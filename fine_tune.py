import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch import optim
from sklearn.metrics import accuracy_score, confusion_matrix
from trainer import create_simple_loader, create_dataset

from config import MODULE_CONFIG, TEST_MODEL_PATH
from networks import EmbeddingNet_cov, EmbeddingNet_res, SiameseNet, TripletNet
import os
from tqdm import tqdm


class FCResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.2):
        super(FCResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out += identity  # 残差连接
        out = self.relu(out)
        return out


class MLPResNetClassifier(nn.Module):
    def __init__(
        self, input_dim, num_classes, hidden_dim=256, num_blocks=2, dropout_rate=0.2
    ):
        super(MLPResNetClassifier, self).__init__()

        # 初始特征提取层
        self.initial_fc = nn.Linear(input_dim, hidden_dim)
        self.initial_bn = nn.BatchNorm1d(hidden_dim)
        self.initial_relu = nn.ReLU()

        # 残差块序列
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.residual_blocks.append(
                FCResidualBlock(hidden_dim, hidden_dim, dropout_rate)
            )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        # 初始特征提取
        x = self.initial_fc(x)
        x = self.initial_bn(x)
        x = self.initial_relu(x)

        # 残差块处理
        for block in self.residual_blocks:
            x = block(x)

        # 分类
        x = self.classifier(x)
        return x


# 定义全连接分类器
class SimpleFCClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleFCClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # 输入层到第一个隐藏层
        self.fc2 = nn.Linear(128, num_classes)  # 隐藏层到输出层
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def check_embeddings_quality(
    train_embeddings, train_labels, val_embeddings, val_labels
):
    """检查嵌入向量和标签的质量

    Args:
        train_embeddings: 训练集嵌入向量
        train_labels: 训练集标签
        val_embeddings: 验证集嵌入向量
        val_labels: 验证集标签

    Returns:
        bool: 数据质量是否合格
    """
    print("\n===== 嵌入向量和标签质量检查 =====")

    # 检查数据形状
    print(f"训练集嵌入向量形状: {train_embeddings.shape}")
    print(f"训练集标签形状: {train_labels.shape}")
    print(f"验证集嵌入向量形状: {val_embeddings.shape}")
    print(f"验证集标签形状: {val_labels.shape}")

    # 检查标签分布
    train_unique_labels = np.unique(train_labels)
    val_unique_labels = np.unique(val_labels)
    print(f"训练集标签类别: {train_unique_labels}")
    print(f"验证集标签类别: {val_unique_labels}")

    # 检查每个类别的样本数量
    # train_start = 0
    # val_start = 0
    for label in train_unique_labels:
        train_count = np.sum(train_labels == label)
        val_count = np.sum(val_labels == label) if label in val_unique_labels else 0
        print(f"类别 {label}: 训练集 {train_count} 样本, 验证集 {val_count} 样本")
        # # 计算类别质心（向量均值）
        # train_centroid = np.mean(
        #     train_embeddings[train_start : train_start + train_count], axis=0
        # )
        # # 计算样本到质心的平均距离作为分散度指标
        # train_dispersion = np.mean(
        #     np.linalg.norm(
        #         train_embeddings[train_start : train_start + train_count]
        #         - train_centroid,
        #         axis=1,
        #     )
        # )
        # print(
        #     f"类别{label}训练集嵌入向量质心(中间10位):{train_centroid[:10]}, 向量维度{train_centroid.shape}, 样本到质心平均距离: {train_dispersion:.4f}"
        # )
        # val_centroid = np.mean(
        #     val_embeddings[val_start : val_start + val_count], axis=0
        # )
        # val_dispersion = np.mean(
        #     np.linalg.norm(
        #         val_embeddings[val_start : val_start + val_count] - val_centroid, axis=1
        #     )
        # )
        # print(
        #     f"类别{label}验证集嵌入向量质心(中间10位):{val_centroid[:10]}, 向量维度{val_centroid.shape}, 样本到质心平均距离: {val_dispersion:.4f}"
        # )

        # train_start += train_count
        # val_start += val_count

    # 检查是否有NaN或无穷大值
    train_has_nan = np.isnan(train_embeddings).any() or np.isinf(train_embeddings).any()
    val_has_nan = np.isnan(val_embeddings).any() or np.isinf(val_embeddings).any()

    if train_has_nan or val_has_nan:
        print("警告: 嵌入向量中存在NaN或无穷大值!")

    print("嵌入向量和标签质量检查通过!")


def train_classifier_with_fc(
    model_path,
    train_embeddings,
    train_labels,
    val_embeddings,
    val_labels,
    epochs=100,
    batch_size=16,
    learning_rate=0.0001,
):
    # 创建自定义Dataset
    class EmbeddingDataset(torch.utils.data.Dataset):
        def __init__(self, embeddings, labels):
            self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.embeddings[idx], self.labels[idx]

    # 创建训练集和验证集的Dataset
    train_dataset = EmbeddingDataset(train_embeddings, train_labels)
    val_dataset = EmbeddingDataset(val_embeddings, val_labels)

    # 创建DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # 定义模型，损失函数和优化器
    input_dim = train_embeddings.shape[1]  # 输入特征的维度
    num_classes = len(np.unique(train_labels))  # 类别数目

    if MODULE_CONFIG["classifier"] == "fc":
        model = SimpleFCClassifier(input_dim, num_classes)
    elif MODULE_CONFIG["classifier"] == "mlp_resnet":
        model = MLPResNetClassifier(input_dim, num_classes)

    # 如果有CUDA，将模型移至GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()  # 分类任务使用交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 添加学习率调度器 - 使用ReduceLROnPlateau，移除verbose参数
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 早停机制
    early_stopping_patience = 15
    best_val_loss = float("inf")
    no_improve_epochs = 0
    best_model_state = None

    # 记录训练过程中的loss
    train_losses = []
    val_losses = []
    accuracies = []

    # 训练模型
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 计算平均训练loss
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()

                # 获取预测结果
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # 计算平均验证loss和准确率
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        accuracy = accuracy_score(all_targets, all_preds)
        accuracies.append(accuracy)

        scheduler.step(avg_val_loss)

        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            no_improve_epochs = 0
            print(f"New best validation loss: {best_val_loss:.4f}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        # 打印每个epoch的结果
        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Accuracy: {accuracy:.4f}"
        )

    # 保存模型
    # 恢复最佳模型状态并保存
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with validation loss: {best_val_loss:.4f}")

    # 绘制loss曲线
    plt.figure(figsize=(12, 4))

    # 绘制loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label="Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.grid(True)

    # 调整子图之间的间距
    plt.tight_layout()

    # 保存图像
    save_path = f"{model_path}_training_curves.png"
    plt.savefig(save_path)
    plt.close()

    # 保存训练历史数据
    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "accuracy": accuracies,
    }
    np.save(f"{model_path}_training_history.npy", history)

    torch.save(model.state_dict(), f"{model_path}_Classifier.pth")
    return model


# 绘制混淆矩阵
def plot_confusion_matrix(model_path, cm, num_classes):
    plt.figure(figsize=(40, 30))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"{i}" for i in range(num_classes)],
        yticklabels=[f"{i}" for i in range(num_classes)],
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    save_path = f"{model_path}_confusion_matrix.jpg"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()  # 关闭图像，释放内存


# 提取嵌入特征和标签
def extract_embeddings(dataloader, model, cuda=torch.cuda.is_available()):
    model.eval()  # 切换为评估模式
    embedding_list = []
    label_list = []

    with torch.no_grad():  # 禁用梯度计算
        for data, target in tqdm(
            dataloader, desc="获取嵌入向量(batch)", total=len(dataloader)
        ):
            # 对于孪生网络，只使用第一个输入（如果 data 为 tuple）
            if isinstance(data, tuple):
                data = data[0]
            elif isinstance(data, list):
                # 如果是列表，我们只使用列表中的第一个tensor，因为模型期望单个tensor
                data = data[0]
            if cuda:
                # 如数据加载器启用了 pin_memory，则使用 non_blocking 可以加速传输
                data = data.cuda(non_blocking=True)

            # 获取嵌入向量（依然保持在 tensor 格式）
            emb = model.get_embedding(data)

            # 收集 tensor 而不是 numpy 数组
            embedding_list.append(emb)
            label_list.append(target)

    # 将所有 batch 的嵌入和标签拼接在一起
    embeddings = torch.cat(embedding_list, dim=0)
    labels = torch.cat(label_list, dim=0)

    # 最后只进行一次 cpu() 和 numpy() 转换
    embeddings = embeddings.cpu().numpy()
    labels = labels.cpu().numpy()

    return embeddings, labels


def fine_tune(classifier_model_path=None):
    model_path = TEST_MODEL_PATH
    # 启用CUDA性能优化
    torch.backends.cudnn.benchmark = True
    # 如果输入大小固定，可以设置
    torch.backends.cudnn.deterministic = True

    # 根据配置选择嵌入网络类型
    if MODULE_CONFIG["embedding_net"] == "cov":
        embedding_net = EmbeddingNet_cov()
    elif MODULE_CONFIG["embedding_net"] == "res":
        embedding_net = EmbeddingNet_res()
    else:
        raise ValueError(
            f"Unknown embedding net type: {MODULE_CONFIG['embedding_net']}"
        )

    if MODULE_CONFIG["dataset"] == "triplet":
        model = TripletNet(embedding_net)
    else:
        model = SiameseNet(embedding_net)
    if torch.cuda.is_available():
        model = model.cuda()

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"加载模型检查点: epoch {checkpoint['epoch']}, 验证损失 {checkpoint['val_loss']:.4f}"
    )
    model_name = os.path.basename(model_path).split(".")[0]

    # 提取训练集和验证集的嵌入特征
    print("正在提取全部嵌入特征...")
    # 定义保存路径
    save_dir = os.path.join("./DATA/Embeddings", os.path.dirname(model_name))
    os.makedirs(save_dir, exist_ok=True)  # 创建保存目录

    # 构建完整的文件路径
    train_embeddings_path = os.path.join(
        save_dir, f"{os.path.basename(model_name)}_train_embeddings.npy"
    )
    train_labels_path = os.path.join(
        save_dir, f"{os.path.basename(model_name)}_train_labels.npy"
    )
    val_embeddings_path = os.path.join(
        save_dir, f"{os.path.basename(model_name)}_val_embeddings.npy"
    )
    val_labels_path = os.path.join(
        save_dir, f"{os.path.basename(model_name)}_val_labels.npy"
    )

    # 如果存在对应的嵌入特征即加载，若不存在则提取
    if os.path.exists(train_embeddings_path):
        print("加载已存在的嵌入特征...")
        train_embeddings = np.load(train_embeddings_path)
        train_labels = np.load(train_labels_path)
        val_embeddings = np.load(val_embeddings_path)
        val_labels = np.load(val_labels_path)
    else:
        print("提取新的嵌入特征...")
        train_dataset, test_dataset = create_dataset()
        train_loader, val_loader = create_simple_loader(train_dataset)
        train_embeddings, train_labels = extract_embeddings(train_loader, model)
        val_embeddings, val_labels = extract_embeddings(val_loader, model)

        # 保存嵌入特征
        print("保存嵌入特征...")
        np.save(train_embeddings_path, train_embeddings)
        np.save(train_labels_path, train_labels)
        np.save(val_embeddings_path, val_embeddings)
        np.save(val_labels_path, val_labels)
    # 查看训练集和验证集是否准确
    # 在提取完嵌入向量后添加检查
    check_embeddings_quality(train_embeddings, train_labels, val_embeddings, val_labels)

    # 训练分类器
    if classifier_model_path is None:
        print("正在训练分类器...")
        classifier = train_classifier_with_fc(
            model_path,
            train_embeddings,
            train_labels,
            val_embeddings,
            val_labels,
            epochs=100,
            batch_size=16,
            learning_rate=0.0003,
        )
    else:
        classifier = SimpleFCClassifier(128, 46)
        classifier.load_state_dict(torch.load(classifier_model_path))
        classifier.cuda()
        print("加载已存在的分类器...")
    eval_classfication(classifier, val_embeddings, val_labels, model_path)


def eval_classfication(classifier, val_embeddings, val_labels, save_path):
    # 评估分类器
    print("正在评估分类器...")
    # 将数据移动到正确的设备上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_embeddings = torch.tensor(val_embeddings, dtype=torch.float32).to(device)
    val_labels = torch.tensor(val_labels, dtype=torch.long).to(device)

    # 使用模型进行预测
    classifier.eval()  # 切换到评估模式
    with torch.no_grad():  # 禁用梯度计算
        # 获取模型输出
        outputs = classifier(val_embeddings)
        # 使用 argmax 获取每个样本的预测类别
        test_predictions = outputs.argmax(dim=1)

    # 计算测试集准确率
    test_accuracy = accuracy_score(
        val_labels.cpu().numpy(), test_predictions.cpu().numpy()
    )
    print(f"测试集准确率: {test_accuracy:.4f}")

    # 绘制混淆矩阵
    cm = confusion_matrix(val_labels.cpu().numpy(), test_predictions.cpu().numpy())
    num_classes = len(np.unique(val_labels.cpu().numpy()))
    plot_confusion_matrix(save_path, cm, num_classes)

    # 输出每类的准确率
    num_classes = cm.shape[0]
    for i in range(num_classes):
        # 计算每个类别的准确率：正确预测的数目 / 该类别的总样本数
        correct_predictions = cm[i, i]
        total_predictions = cm[i, :].sum()  # 该类总预测样本数
        class_accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0
        )
        print(f"类 {i} 的准确率: {class_accuracy * 100:.2f}%")


if __name__ == "__main__":
    fine_tune()
    # fine_tune(classifier_model_path=r".\models\best_model.pth_Classifier.pth")
