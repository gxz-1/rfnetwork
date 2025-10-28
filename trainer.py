import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import (
    SiameseIQDataset_random,
    SiameseIQDataset_balanced_1,
    SiameseIQDataset_balanced_2,
    TripletIQDataset,
    IQDataset,
)
from networks import EmbeddingNet_cov, EmbeddingNet_res, SiameseNet, TripletNet
from losses import ContrastiveLoss, TripletLoss
import os
from utils import visualize_embeddings
from config import (
    DATA_CONFIG,
    TRAIN_CONFIG,
    LOSS_CONFIG,
    MODULE_CONFIG,
    TEST_MODEL_PATH,
    CHECKPOINT_PATH,
)
import random
import copy
import pickle
import matplotlib.pyplot as plt


def fit(
    train_loader,
    val_loader,
    model,
    loss_fn,
    optimizer,
    scheduler,
    n_epochs,
    cuda,
    log_interval,
    metrics=[],
    start_epoch=0,
):
    """
    功能：训练过程管理
    参数：
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        model: 模型
        loss_fn: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        n_epochs: 训练轮数
        cuda: 是否使用GPU
        log_interval: 日志打印间隔
        metrics: 评估指标列表
        start_epoch: 开始训练的轮数
    加载器、模型、损失函数和指标应该一起工作，即模型应该能够处理加载器的输出数据，
    损失函数应该处理加载器的输出目标，模型输出应该与损失函数和指标的输出兼容

    示例：分类：批量加载器、分类模型、NLL损失、准确率指标
    孪生网络：孪生加载器、孪生模型、对比损失
    在线三元组学习：批量加载器、嵌入模型、在线三元组损失
    """
    best_val_loss = float("inf")
    # 添加列表用于记录训练和验证损失
    train_losses = []
    val_losses = []

    for epoch in range(0, start_epoch):
        optimizer.step()  # 先更新参数
        scheduler.step()  # 再更新学习率

    for epoch in range(start_epoch, n_epochs):
        # Train stage
        train_loss, metrics = train_epoch(
            train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics
        )
        # 记录训练损失
        train_losses.append(train_loss)

        message = "Epoch: {}/{}. Train set: Average loss: {:.4f}".format(
            epoch + 1, n_epochs, train_loss
        )
        for metric in metrics:
            message += "\t{}: {}".format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)
        # 记录验证损失
        val_losses.append(val_loss)

        message += "\nEpoch: {}/{}. Validation set: Average loss: {:.4f}".format(
            epoch + 1, n_epochs, val_loss
        )
        for metric in metrics:
            message += "\t{}: {}".format(metric.name(), metric.value())

        print(message)

        optimizer.step()  # 先更新参数
        scheduler.step()  # 再更新学习率

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                "models/best_model.pth",
            )

        # 定期保存检查点（每N个epoch）
        if (epoch + 1) % TRAIN_CONFIG["save_interval"] == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                f"models/checkpoint_epoch_{epoch+1}.pth",
            )

    # 训练结束后绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Val Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # 保存图像
    plt.savefig("models/training_loss_curves.png")
    plt.close()

    # 保存训练历史数据
    history = {"train_loss": train_losses, "val_loss": val_losses}
    np.save("models/training_history.npy", history)


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx == 0:
            print("\n第一个训练批次数据检查:")
            print(f"输入数据类型: {type(data)}")
            print(f"输入数据形状: {[d.shape for d in data]}")
            print(f"输入数据范围: {[f'[{d.min():.4f}, {d.max():.4f}]' for d in data]}")
            print(f"标签示例: {target[:5]}")
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()

        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = "Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                batch_idx * len(data[0]),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                np.mean(losses),
            )
            for metric in metrics:
                message += "\t{}: {}".format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= batch_idx + 1
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None and len(target) > 0:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None and len(target) > 0:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = (
                loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            )
            val_loss += loss.item()

            for metric in metrics:
                if target is not None and len(target) > 0:
                    metric(outputs, target, loss_outputs)
                else:
                    metric(outputs, None, loss_outputs)

    return val_loss, metrics


def create_simple_loader(dataset):
    """
    创建用于测试的数据加载器
    """
    # 创建通用的数据集参数字典，复用dataset的参数
    dataset_params = {
        "data_dir": DATA_CONFIG["data_dir"],
        "class_num": DATA_CONFIG["class_num"],
        "samples_per_frame": DATA_CONFIG["samples_per_frame"],
        "samples_length": DATA_CONFIG["samples_length"],
        "stride": DATA_CONFIG["stride"],
        "cache_size": DATA_CONFIG["cache_size"],
        "normalize": DATA_CONFIG["normalize"],
        "train_ratio": DATA_CONFIG["train_ratio"],
        "split_seed": DATA_CONFIG["split_seed"],
        "dataset_type": DATA_CONFIG["dataset_type"],
        "normalize_way": DATA_CONFIG["normalize_way"],
    }

    # 使用IQDataset而不是孪生数据集
    train_dataset = IQDataset(**dataset_params)
    val_dataset = copy.deepcopy(train_dataset)
    val_dataset.change_split_mode("val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=DATA_CONFIG["batch_size"],
        shuffle=False,
        num_workers=DATA_CONFIG["num_workers"],
        pin_memory=DATA_CONFIG["pin_memory"],
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=DATA_CONFIG["batch_size"],
        shuffle=False,
        num_workers=DATA_CONFIG["num_workers"],
        pin_memory=DATA_CONFIG["pin_memory"],
        drop_last=True
    )

    return train_loader, val_loader


def resume_training(checkpoint_path, model, optimizer, scheduler):
    """
    加载检查点并恢复训练状态

    参数:
        checkpoint_path: 检查点文件路径
        model: 模型实例
        optimizer: 优化器实例
        scheduler: 学习率调度器实例

    返回:
        start_epoch: 继续训练的起始轮次
        best_val_loss: 最佳验证损失
    """
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"检查点文件 {checkpoint_path} 不存在")
        return 0, float("inf")

    # 加载检查点
    checkpoint = torch.load(checkpoint_path)

    # 恢复模型权重
    model.load_state_dict(checkpoint["model_state_dict"])

    # 恢复优化器状态
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 获取恢复的轮次和损失
    start_epoch = checkpoint["epoch"] + 1  # 从下一轮开始
    best_val_loss = checkpoint.get("val_loss", float("inf"))

    print(
        f"从检查点恢复训练: epoch {checkpoint['epoch']}, 验证损失 {best_val_loss:.4f}"
    )

    return start_epoch, best_val_loss


def create_dataset():
    """
    创建或加载数据集实例
    如果存在已保存的数据集文件，则直接加载；否则创建新的数据集实例并保存
    """
    # 创建保存数据集的目录
    dataset_cache_dir = os.path.join("DATA", "dataset_cache")
    os.makedirs(dataset_cache_dir, exist_ok=True)

    # 生成缓存文件名，包含关键参数信息
    cache_filename = (
        f"dataset_c{DATA_CONFIG['class_num']}"
        f"_s{DATA_CONFIG['samples_per_frame']}"
        f"_l{DATA_CONFIG['samples_length']}"
        f"_stride{DATA_CONFIG['stride']}"
        f"_r{int(DATA_CONFIG['train_ratio']*100)}"
        f"_seed{DATA_CONFIG['split_seed']}"
        f"_{MODULE_CONFIG['dataset']}"
        f"_{DATA_CONFIG['dataset_type']}.pkl"
    )
    cache_path = os.path.join(dataset_cache_dir, cache_filename)

    # 尝试加载缓存的数据集
    if os.path.exists(cache_path):
        try:
            print(f"尝试加载缓存的数据集: {cache_path}")
            with open(cache_path, "rb") as f:
                train_dataset = pickle.load(f)
                val_dataset = pickle.load(f)
            print("成功加载缓存的数据集")
            val_dataset.change_split_mode("val")
            return train_dataset, val_dataset
        except Exception as e:
            print(f"加载缓存数据集失败: {str(e)}")
            print("将创建新的数据集实例")
            # 如果加载失败，删除可能损坏的缓存文件
            try:
                os.remove(cache_path)
                print(f"已删除损坏的缓存文件: {cache_path}")
            except OSError as e:
                print(f"删除缓存文件失败: {str(e)}")

    # 如果没有缓存或加载失败，创建新的数据集实例
    # 创建通用的数据集参数字典
    dataset_params = {
        "data_dir": DATA_CONFIG["data_dir"],
        "class_num": DATA_CONFIG["class_num"],
        "samples_per_frame": DATA_CONFIG["samples_per_frame"],
        "samples_length": DATA_CONFIG["samples_length"],
        "stride": DATA_CONFIG["stride"],
        "cache_size": DATA_CONFIG["cache_size"],
        "normalize": DATA_CONFIG["normalize"],
        "train_ratio": DATA_CONFIG["train_ratio"],
        "split_seed": DATA_CONFIG["split_seed"],
        "dataset_type": DATA_CONFIG["dataset_type"],
        "normalize_way": DATA_CONFIG["normalize_way"],
    }

    # 使用字典映射来创建数据集
    dataset_map = {
        "random": SiameseIQDataset_random,
        "balanced_1": SiameseIQDataset_balanced_1,
        "balanced_2": SiameseIQDataset_balanced_2,
        "triplet": TripletIQDataset,
        "simple": IQDataset,
    }

    # 创建训练集和验证集实例
    print("创建新的数据集实例...")
    train_dataset = dataset_map[MODULE_CONFIG["dataset"]](
        **dataset_params, split_mode="train"
    )
    val_dataset = dataset_map[MODULE_CONFIG["dataset"]](
        **dataset_params, split_mode="val"
    )

    # 保存数据集实例到缓存
    try:
        print(f"保存数据集到缓存: {cache_path}")
        # 确保目录存在
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        # 使用临时文件保存，避免保存过程中的中断导致文件损坏
        temp_path = cache_path + ".tmp"
        with open(temp_path, "wb") as f:
            pickle.dump(train_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(val_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        # 保存成功后，将临时文件重命名为正式文件
        os.replace(temp_path, cache_path)
        print("数据集缓存保存成功")
    except Exception as e:
        print(f"保存数据集缓存失败: {str(e)}")
        # 清理临时文件
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError as e:
            print(f"清理临时文件失败: {str(e)}")

    return train_dataset, val_dataset


def main(model_path=None, resume=False):
    if model_path is None and resume:
        print("未指定模型路径，不可继续训练")
        return

    # 应用CUDA配置
    torch.backends.cudnn.benchmark = TRAIN_CONFIG["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = TRAIN_CONFIG["cudnn_deterministic"]

    train_dataset, val_dataset = create_dataset()
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"训练集模式: {train_dataset.split_mode}")
    print(f"验证集模式: {val_dataset.split_mode}")
    # 创建数据加载器，使用DATA_CONFIG中的参数
    train_loader = DataLoader(
        train_dataset,
        batch_size=DATA_CONFIG["batch_size"],
        shuffle=DATA_CONFIG["shuffle"],
        num_workers=DATA_CONFIG["num_workers"],
        pin_memory=DATA_CONFIG["pin_memory"],
        drop_last=True,
        **({"prefetch_factor": DATA_CONFIG["prefetch_factor"]} if DATA_CONFIG["num_workers"] > 0 and DATA_CONFIG["prefetch_factor"] > 0 else {})
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=DATA_CONFIG["batch_size"],
        shuffle=DATA_CONFIG["shuffle"],
        num_workers=DATA_CONFIG["num_workers"],
        pin_memory=DATA_CONFIG["pin_memory"],
        drop_last=True,
        **({"prefetch_factor": DATA_CONFIG["prefetch_factor"]} if DATA_CONFIG["num_workers"] > 0 and DATA_CONFIG["prefetch_factor"] > 0 else {})
    )

    # 创建嵌入网络，使用MODEL_CONFIG中的参数
    if MODULE_CONFIG["embedding_net"] == "cov":
        embedding_net = EmbeddingNet_cov()
    elif MODULE_CONFIG["embedding_net"] == "res":
        embedding_net = EmbeddingNet_res()

    # 定义损失函数和优化器，使用LOSS_CONFIG和TRAIN_CONFIG中的参数
    if MODULE_CONFIG["dataset"] == "triplet":
        loss_fn = TripletLoss(margin=LOSS_CONFIG["triplet_margin"])
        model = TripletNet(embedding_net)
    else:
        loss_fn = ContrastiveLoss(margin=LOSS_CONFIG["contrastive_margin"])
        model = SiameseNet(embedding_net)
    if MODULE_CONFIG["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=TRAIN_CONFIG["learning_rate"]
        )
    elif MODULE_CONFIG["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=TRAIN_CONFIG["learning_rate"]
        )

    # 定义学习率调度器
    if MODULE_CONFIG["scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            TRAIN_CONFIG["scheduler_step_size"],
            gamma=TRAIN_CONFIG["scheduler_gamma"],
            last_epoch=-1,
        )
    elif MODULE_CONFIG["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=TRAIN_CONFIG["n_epochs"],
            last_epoch=-1,
        )

    # 使用TRAIN_CONFIG中的训练参数
    n_epochs = TRAIN_CONFIG["n_epochs"]
    log_interval = TRAIN_CONFIG["log_interval"]
    start_epoch = 0

    # 检查CUDA可用性
    cuda = TRAIN_CONFIG["cuda"] and torch.cuda.is_available()
    if cuda:
        model = model.cuda()
        print("使用CUDA运行")

    # 加载已有模型进行测试
    if model_path is not None and not resume:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"加载模型检查点: epoch {checkpoint['epoch']}, 验证损失 {checkpoint['val_loss']:.4f}"
        )
        # 创建simple数据加载器
        train_loader, val_loader = create_simple_loader(train_dataset)

        # 可视化嵌入向量
        print("正在可视化嵌入向量...")
        visualize_embeddings(
            train_loader,
            val_loader,
            model,
            num_classes=DATA_CONFIG["class_num"],
            cuda=cuda,
        )
    else:
        print("训练模型...")
        # 从检查点恢复训练，或者从头开始训练
        if resume:
            # 加载检查点
            start_epoch, best_val_loss = resume_training(
                CHECKPOINT_PATH, model, optimizer, scheduler
            )
        # 可以改为直接调用fit函数
        fit(
            train_loader,
            val_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            n_epochs,
            cuda=cuda,
            log_interval=log_interval,
            start_epoch=start_epoch,
        )


def validate_data(show_detail=False):
    """
    验证数据集的划分是否正确，检查以下几个方面：
    1. 训练集和验证集的大小是否符合设定的比例
    2. 训练集和验证集是否不存在重叠
    3. 每个类别的样本是否按照相同的比例划分
    4. 索引划分是否正确保存和加载
    """
    # 获取训练集和验证集
    train_dataset, val_dataset = create_dataset()

    # 1. 检查训练集和验证集大小及比例
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    total_size = train_size + val_size
    actual_ratio = train_size / total_size

    print("\n数据集验证报告:")
    print("----------------------------")
    print(f"总样本数: {total_size}")
    print(f"训练集样本数: {train_size}")
    print(f"验证集样本数: {val_size}")
    print(
        f"实际训练集比例: {actual_ratio:.4f}, 配置比例: {DATA_CONFIG['train_ratio']:.4f}"
    )

    # 2. 检查训练集和验证集是否不存在重叠
    train_indices_set = set(train_dataset.indices)
    val_indices_set = set(val_dataset.indices)
    overlap = train_indices_set.intersection(val_indices_set)

    if len(overlap) == 0:
        print("✓ 训练集和验证集没有重叠")
    else:
        print(f"✗ 训练集和验证集存在 {len(overlap)} 个重叠样本")

    # 3. 检查每个类别的划分比例
    print("\n按类别划分验证:")
    print("----------------------------")
    class_stats = []
    if train_dataset.dataset_type == "single":
        num_classes = train_dataset.class_num
        for class_idx in range(num_classes):
            # 计算该类别在训练集和验证集中的样本数
            train_class_samples = sum(
                1
                for idx in train_dataset.indices
                if train_dataset._find_file_and_sample(idx)[0] == class_idx
            )
            val_class_samples = sum(
                1
                for idx in val_dataset.indices
                if val_dataset._find_file_and_sample(idx)[0] == class_idx
            )
            class_total = train_class_samples + val_class_samples
            class_ratio = train_class_samples / class_total if class_total > 0 else 0

            class_stats.append(
                {
                    "class_idx": class_idx,
                    "train_samples": train_class_samples,
                    "val_samples": val_class_samples,
                    "total": class_total,
                    "ratio": class_ratio,
                }
            )

            print(
                f"类别 {train_dataset.data_files[class_idx]}（{class_idx}）: 训练集 {train_class_samples}，验证集 {val_class_samples}，"
                f"总数 {class_total}，训练比例 {class_ratio:.4f}"
            )
    elif train_dataset.dataset_type == "recursive":
        # 对于递归目录结构，使用file_to_class映射来统计每个类别的样本
        num_classes = len(set(train_dataset.file_to_class.values()))
        for class_idx in range(num_classes):
            # 计算该类别在训练集和验证集中的样本数
            train_class_samples = sum(
                1
                for idx in train_dataset.indices
                if train_dataset.file_to_class[
                    train_dataset.data_files[
                        train_dataset._find_file_and_sample(idx)[0]
                    ]
                ]
                == class_idx
            )
            val_class_samples = sum(
                1
                for idx in val_dataset.indices
                if val_dataset.file_to_class[
                    val_dataset.data_files[val_dataset._find_file_and_sample(idx)[0]]
                ]
                == class_idx
            )
            class_total = train_class_samples + val_class_samples
            class_ratio = train_class_samples / class_total if class_total > 0 else 0

            # 获取类别目录名
            class_dir = train_dataset.class_dirs[class_idx]

            class_stats.append(
                {
                    "class_idx": class_idx,
                    "train_samples": train_class_samples,
                    "val_samples": val_class_samples,
                    "total": class_total,
                    "ratio": class_ratio,
                }
            )

            print(
                f"类别 {class_dir}（{class_idx}）: 训练集 {train_class_samples}，验证集 {val_class_samples}，"
                f"总数 {class_total}，训练比例 {class_ratio:.4f}"
            )
            if show_detail:
                # 额外显示该类别下的文件分布
                class_files = [
                    f for f, c in train_dataset.file_to_class.items() if c == class_idx
                ]
                print(f"  文件列表（共{len(class_files)}个）:")
                for file_path in class_files:
                    file_train_samples = sum(
                        1
                        for idx in train_dataset.indices
                        if train_dataset.data_files[
                            train_dataset._find_file_and_sample(idx)[0]
                        ]
                        == file_path
                    )
                    file_val_samples = sum(
                        1
                        for idx in val_dataset.indices
                        if val_dataset.data_files[
                            val_dataset._find_file_and_sample(idx)[0]
                        ]
                        == file_path
                    )
                    print(
                        f"    - {file_path}: 训练集 {file_train_samples}，验证集 {file_val_samples}"
                    )

    # 检查所有类别的划分比例是否一致
    ratios = [stat["ratio"] for stat in class_stats]
    max_ratio_diff = max(ratios) - min(ratios) if ratios else 0

    if max_ratio_diff < 0.01:
        print(f"✓ 所有类别的训练比例一致，最大差异: {max_ratio_diff:.4f}")
    else:
        print(f"✗ 类别之间的训练比例不一致，最大差异: {max_ratio_diff:.4f}")

    # 4. 验证索引文件是否正确保存和加载
    split_dir = os.path.join("DATA", "splits")
    split_file = os.path.join(
        split_dir,
        f"split_c{train_dataset.class_num}_r{int(train_dataset.train_ratio*100)}_s{train_dataset.split_seed}.npy",
    )

    if os.path.exists(split_file):
        # 加载保存的索引
        saved_indices = np.load(split_file, allow_pickle=True).item()
        saved_train_indices = set(saved_indices["train"])
        saved_val_indices = set(saved_indices["val"])

        # 比较与当前使用的索引是否一致
        current_train_indices = set(train_dataset.indices)
        current_val_indices = set(val_dataset.indices)

        train_match = saved_train_indices == current_train_indices
        val_match = saved_val_indices == current_val_indices

        if train_match and val_match:
            print(f"✓ 索引文件正确加载: {split_file}")
        else:
            print("✗ 索引文件与当前使用的索引不一致")
            if not train_match:
                print("  - 训练集索引不匹配")
            if not val_match:
                print("  - 验证集索引不匹配")
    else:
        print(f"✗ 索引文件不存在: {split_file}")

    # 数据样本检查
    print("\n数据样本检查:")
    print("----------------------------")

    # 随机检查一些样本
    # 从训练集中随机抽取5个样本
    sample_indices = random.sample(
        range(len(train_dataset)), min(5, len(train_dataset))
    )

    for i, idx in enumerate(sample_indices):
        sample, label = train_dataset[idx]
        print(f"训练集样本 {i+1}:")

        # 检查样本是否为元组或列表（如孪生或三元组网络的情况）
        if isinstance(sample, (tuple, list)):
            print(f"  - 样本类型: {'三元组' if len(sample) == 3 else '孪生'}")
            for j, tensor in enumerate(sample):
                print(f"  - 子样本 {j+1} 形状: {tensor.shape}")
                print(
                    f"  - 子样本 {j+1} 数据: 通道1前10位: {[f'{x:.3f}' for x in tensor[0, :10].tolist()]}\n, 通道2前10位: {[f'{x:.3f}' for x in tensor[1, :10].tolist()]}"
                )

        else:
            # 单个样本的情况
            print(f"  - 形状: {sample.shape}")
            print(
                f"  - 子样本 {j+1} 数据: 通道1前10位: {[f'{x:.3f}' for x in tensor[0, :10].tolist()]}\n, 通道2前10位: {[f'{x:.3f}' for x in tensor[1, :10].tolist()]}"
            )

        print(f"  - 标签(类别): {label}")

    print("\n验证完成!")


def process_iq_file_to_npy(dir_path):
    """
    读取一个.dat文件中的IQ数据，并保存为.npy格式的复数数组。

    参数:
    dir_path (str): .dat 文件的完整路径

    输出:
    保存为同名 .npy 文件
    """
    # Step 1: 读取 .dat 文件内容为 int32
    data = np.fromfile(dir_path, dtype=np.int32)

    # Step 2: 转为无符号 uint32 并格式化为 8 位十六进制字符串
    data_uint32 = data.astype(np.uint32)
    hex_strs = np.vectorize(lambda x: f"{x:08x}")(data_uint32)

    # Step 3: 提取 I（前4位）和 Q（后4位）部分
    data_I = np.array([int(h[0:4], 16) for h in hex_strs], dtype=np.int32)
    data_Q = np.array([int(h[4:8], 16) for h in hex_strs], dtype=np.int32)

    # Step 4: 处理补码（符号扩展）
    data_I[data_I > 32768] -= 65536
    data_Q[data_Q > 32768] -= 65536

    # Step 5: 合成为复数数组
    data_complex = data_I + 1j * data_Q

    # Step 6: 保存为 .npy 文件
    npy_path = os.path.splitext(dir_path)[0] + ".npy"
    np.save(npy_path, data_complex)


if __name__ == "__main__":
    status = MODULE_CONFIG["status"]
    if status == 1:
        # 测试模型
        main(model_path=TEST_MODEL_PATH)
    elif status == 2:
        # 从头训练
        main()
    elif status == 3:
        # 继续训练
        main(model_path=CHECKPOINT_PATH, resume=True)
    elif status == 4:
        # 验证数据
        validate_data()
    # input_dir = r"./DATA/tmp"
    # for root, dirs, files in os.walk(input_dir):
    #     for file in files:
    #         if file.endswith(".dat"):
    #             file_path = os.path.join(root, file)
    #             process_iq_file_to_npy(file_path)
