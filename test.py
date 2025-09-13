import torch
import time
from torch.utils.data import DataLoader
from datasets import IQDataset
from networks import EmbeddingNet_res, EmbeddingNet_cov, TripletNet, SiameseNet
from config import DATA_CONFIG
import os
import random
from fine_tune import SimpleFCClassifier, MLPResNetClassifier
from config import MODULE_CONFIG


def load_models(embedding_model_path, classifier_model_path):
    """
    加载嵌入模型和分类模型

    参数:
        embedding_model_path: 嵌入模型路径
        classifier_model_path: 分类模型路径

    返回:
        embedding_model: 嵌入模型
        classifier_model: 分类模型
    """
    # 创建模型实例
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
        embedding_model = TripletNet(embedding_net)
    else:
        embedding_model = SiameseNet(embedding_net)

    if MODULE_CONFIG["classifier"] == "fc":
        classifier_model = SimpleFCClassifier(128, num_classes=40)
    elif MODULE_CONFIG["classifier"] == "mlp_resnet":
        classifier_model = MLPResNetClassifier(128, num_classes=40)

    # 加载模型权重
    embedding_checkpoint = torch.load(embedding_model_path, map_location="cpu")
    classifier_checkpoint = torch.load(classifier_model_path, map_location="cpu")

    # 加载嵌入模型权重
    if (
        isinstance(embedding_checkpoint, dict)
        and "model_state_dict" in embedding_checkpoint
    ):
        embedding_model.load_state_dict(embedding_checkpoint["model_state_dict"])
    else:
        embedding_model.load_state_dict(embedding_checkpoint)

    # 加载分类模型权重
    if (
        isinstance(classifier_checkpoint, dict)
        and "model_state_dict" in classifier_checkpoint
    ):
        classifier_model.load_state_dict(classifier_checkpoint["model_state_dict"])
    else:
        classifier_model.load_state_dict(classifier_checkpoint)

    # 设置为评估模式
    embedding_model.eval()
    classifier_model.eval()

    return embedding_model, classifier_model


def create_test_loader(num_samples=60):
    """
    创建测试数据加载器，随机选择指定数量的样本

    参数:
        num_samples: 需要选择的样本数量

    返回:
        test_loader: 测试数据加载器
    """
    # 创建数据集实例
    dataset_params = {
        "data_dir": DATA_CONFIG["data_dir"],
        "class_num": DATA_CONFIG["class_num"],
        "samples_per_frame": DATA_CONFIG["samples_per_frame"],
        "samples_length": DATA_CONFIG["samples_length"],
        "stride": DATA_CONFIG["stride"],
        "cache_size": DATA_CONFIG["cache_size"],
        "normalize": DATA_CONFIG["normalize"],
        "preprocess": DATA_CONFIG["preprocess"],
        "train_ratio": DATA_CONFIG["train_ratio"],
        "split_seed": DATA_CONFIG["split_seed"],
        "dataset_type": DATA_CONFIG["dataset_type"],
        "normalize_way": DATA_CONFIG["normalize_way"],
    }

    # 创建数据集并设置为验证模式
    test_dataset = IQDataset(**dataset_params, split_mode="val")

    # 随机选择指定数量的样本
    indices = random.sample(range(len(test_dataset)), num_samples)
    test_dataset = torch.utils.data.Subset(test_dataset, indices)

    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 单样本推理
        shuffle=False,
        num_workers=0,  # 单进程加载
        pin_memory=False,
    )

    return test_loader


def test_model_performance(embedding_model, classifier_model, test_loader):
    """
    测试模型性能，计算平均推理时间

    参数:
        embedding_model: 嵌入模型
        classifier_model: 分类模型
        test_loader: 测试数据加载器

    返回:
        avg_time: 平均推理时间（毫秒）
        correct: 正确预测的样本数
        total: 总样本数
    """
    total_time = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            # 记录开始时间
            start_time = time.time()

            # 通过嵌入模型
            embedding = embedding_model.get_embedding(data)

            # 通过分类模型
            output = classifier_model(embedding)

            # 计算预测结果
            pred = output.argmax(dim=1)

            # 记录结束时间
            end_time = time.time()

            # 累加时间（转换为毫秒）
            total_time += (end_time - start_time) * 1000

            # 统计正确预测数
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    # 计算平均时间
    avg_time = total_time / len(test_loader)

    return avg_time, correct, total


def main():
    # 设置随机种子以确保可重复性
    random.seed(42)
    torch.manual_seed(42)

    # 模型路径
    embedding_model_path = "models/best_model.pth"
    classifier_model_path = "models/best_model.pth_Classifier.pth"

    # 检查模型文件是否存在
    if not os.path.exists(embedding_model_path):
        raise FileNotFoundError(f"嵌入模型文件不存在: {embedding_model_path}")
    if not os.path.exists(classifier_model_path):
        raise FileNotFoundError(f"分类模型文件不存在: {classifier_model_path}")

    print("正在加载模型...")
    embedding_model, classifier_model = load_models(
        embedding_model_path, classifier_model_path
    )

    print("正在准备测试数据...")
    test_loader = create_test_loader(num_samples=60)

    print("开始性能测试...")
    avg_time, correct, total = test_model_performance(
        embedding_model, classifier_model, test_loader
    )

    # 打印测试结果
    print("\n测试结果:")
    print(f"平均推理时间: {avg_time:.2f} 毫秒")
    print(f"准确率: {correct}/{total} ({100. * correct / total:.2f}%)")


if __name__ == "__main__":
    main()
