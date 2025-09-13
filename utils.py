from itertools import combinations

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import h5py
import scipy.io as sio
import os
from tqdm import tqdm


def pdist(vectors):
    distance_matrix = (
        -2 * vectors.mm(torch.t(vectors))
        + vectors.pow(2).sum(dim=1).view(1, -1)
        + vectors.pow(2).sum(dim=1).view(-1, 1)
    )
    return distance_matrix


# 未使用的类 - 用于选择正负样本对的基类
class PairSelector:
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class HardNegativePairSelector(PairSelector):
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """

    def __init__(self, cpu=True):
        super(HardNegativePairSelector, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)

        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[
            (labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()
        ]
        negative_pairs = all_pairs[
            (labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()
        ]

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[
            : len(positive_pairs)
        ]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = labels == label
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(
                combinations(label_indices, 2)
            )  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [
                [anchor_positive[0], anchor_positive[1], neg_ind]
                for anchor_positive in anchor_positives
                for neg_ind in negative_indices
            ]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(
        np.logical_and(loss_values < margin, loss_values > 0)
    )[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = labels == label
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(
                combinations(label_indices, 2)
            )  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[
                anchor_positives[:, 0], anchor_positives[:, 1]
            ]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = (
                    ap_distance
                    - distance_matrix[
                        torch.LongTensor(np.array([anchor_positive[0]])),
                        torch.LongTensor(negative_indices),
                    ]
                    + self.margin
                )
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append(
                        [anchor_positive[0], anchor_positive[1], hard_negative]
                    )

        if len(triplets) == 0:
            triplets.append(
                [anchor_positive[0], anchor_positive[1], negative_indices[0]]
            )

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(
        margin=margin, negative_selection_fn=hardest_negative, cpu=cpu
    )


def RandomNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(
        margin=margin, negative_selection_fn=random_hard_negative, cpu=cpu
    )


def SemihardNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(
        margin=margin,
        negative_selection_fn=lambda x: semihard_negative(x, margin),
        cpu=cpu,
    )


def extract_embeddings(dataloader, model, cuda=True):
    """
    从数据加载器中提取模型的嵌入向量
    """
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for data, target in dataloader:
            if isinstance(data, tuple):
                # 对于孪生网络，我们只使用第一个输入
                data = data[0]
            if cuda:
                data = data.cuda()

            # 获取嵌入向量
            emb = model.get_embedding(data)
            embeddings.append(emb.cpu().numpy())
            labels.extend(target.numpy())

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    return embeddings, labels


def plot_embeddings(
    embeddings, labels, num_classes=None, figsize=(10, 8), save_path=None
):
    """
    使用t-SNE可视化嵌入向量

    参数:
        embeddings: 嵌入向量数组
        labels: 标签数组
        num_classes: 类别数量（如果为None，则自动从标签中获取）
        figsize: 图像大小
    """
    # 如果未指定类别数量，则从标签中获取
    if num_classes is None:
        num_classes = len(np.unique(labels))

    # 使用t-SNE降维到2D
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        early_exaggeration=12,
        learning_rate=200,
        n_iter=1000,
        random_state=42,
    )
    embeddings_2d = tsne.fit_transform(embeddings)

    # 创建图形
    plt.figure(figsize=figsize)

    # 设置颜色映射
    colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))

    # 绘制每个类别的散点图
    for i in range(num_classes):
        mask = labels == i
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=f"Class {i}",
            alpha=0.6,
        )

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("t-SNE visualization of embeddings")
    plt.tight_layout()

    # 保存图像
    if save_path:
        plt.savefig(f"{num_classes}_{save_path}", bbox_inches="tight", dpi=300)
    else:
        plt.savefig(
            f"embeddings_visualization_{num_classes}classes.png",
            bbox_inches="tight",
            dpi=300,
        )
    plt.close()


def visualize_embeddings(train_loader, val_loader, model, num_classes=None, cuda=True):
    """
    提取并可视化训练集和验证集的嵌入向量

    参数:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        model: 训练好的模型
        num_classes: 类别数量
        cuda: 是否使用GPU
    """
    print("提取训练集嵌入向量...")
    train_embeddings, train_labels = extract_embeddings(train_loader, model, cuda)
    print("可视化训练集嵌入向量...")
    plot_embeddings(train_embeddings, train_labels, num_classes, save_path="train.png")

    print("提取验证集嵌入向量...")
    val_embeddings, val_labels = extract_embeddings(val_loader, model, cuda)
    print("可视化验证集嵌入向量...")
    plot_embeddings(val_embeddings, val_labels, num_classes, save_path="val.png")

    return (train_embeddings, train_labels), (val_embeddings, val_labels)


def process_iq_mat_to_npy(dir_path, output_path):
    """
    递归读取给定目录下所有的.mat文件，将IQ数据保存为.npy文件。

    参数:
    dir_path (str): 要处理的目录路径
    output_path (str): 输出目录路径

    说明:
    - 递归遍历目录下的所有.mat文件
    - 读取.mat文件中的IQ数据（支持MATLAB v7.3格式）
    - 将数据保存到output_path中，保持原有的目录结构
    - 文件扩展名从.mat改为.npy
    """

    def process_single_file(mat_path):
        """处理单个.mat文件"""
        try:
            # 尝试使用scipy.io读取（适用于MATLAB v7.0及以下版本）
            try:
                mat_data = sio.loadmat(mat_path)
                # 获取IQ数据
                if "enhancedata" in mat_data:
                    iq_data = mat_data["enhancedata"]
                    # 将嵌套的ndarray转换为直接的复数数组
                    iq_data = np.array(
                        [x.item() for x in iq_data.flatten()], dtype=np.complex64
                    )
                else:
                    # 如果找不到'data'字段，尝试获取第一个数组
                    for key in mat_data.keys():
                        if not key.startswith("__"):  # 跳过mat文件的内置字段
                            iq_data = mat_data[key]
                            break
                    else:
                        print(f"警告: 在文件 {mat_path} 中未找到IQ数据")
                        return
            except NotImplementedError:
                # 如果是MATLAB v7.3格式，使用h5py读取
                with h5py.File(mat_path, "r") as f:
                    # 读取数据
                    data = np.array(f["enhancedata"])

                    # 检查数据类型
                    if (
                        data.dtype.names is not None
                        and "real" in data.dtype.names
                        and "imag" in data.dtype.names
                    ):
                        # 如果是结构化数组，包含real和imag字段
                        iq_data = data["real"] + 1j * data["imag"]
                    else:
                        # 如果是普通数组，直接使用
                        iq_data = data

            # 确保数据是复数类型
            if not np.iscomplexobj(iq_data):
                print(f"警告: 文件 {mat_path} 中的数据不是复数类型")
                return

            # 计算相对路径
            rel_path = os.path.relpath(mat_path, dir_path)
            # 将扩展名从.mat改为.npy
            rel_path = os.path.splitext(rel_path)[0] + ".npy"
            # 构建完整的输出路径
            npy_path = os.path.join(output_path, rel_path)

            # 确保输出目录存在
            os.makedirs(os.path.dirname(npy_path), exist_ok=True)

            # 保存为.npy文件
            np.save(npy_path, iq_data)

        except Exception as e:
            print(f"处理文件 {mat_path} 时出错: {str(e)}")

    # 获取所有.mat文件
    mat_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".mat"):
                mat_files.append(os.path.join(root, file))

    if not mat_files:
        print(f"在目录 {dir_path} 中未找到.mat文件")
        return

    print(f"找到 {len(mat_files)} 个.mat文件，开始处理...")

    # 使用tqdm显示进度
    for mat_file in tqdm(mat_files, desc="处理.mat文件"):
        process_single_file(mat_file)

    print("处理完成！")


# 示例使用方法
if __name__ == "__main__":
    # 假设已经有训练好的模型和数据加载器
    # visualize_embeddings(train_loader, val_loader, model, num_classes=5)
    dir1 = r"./DATA/IQ数据卫星46分类/SNR30_FS6000/暂存_Enhanced"
    dir2 = r"./DATA/IQ数据卫星46分类/SNR30_FS6000/中星12_Enhanced"
    dir3 = r"./DATA/IQ数据卫星46分类/SNR30_FS6000/中星10_Enhanced"
    output_dir = r"./DATA/preprocessed"
    dirs = [dir1, dir2, dir3]
    for dir_path in dirs:
        process_iq_mat_to_npy(dir_path, output_dir)
