import torch
import time
from torch.utils.data import DataLoader
from datasets import IQDataset
from networks import EmbeddingNet_res, EmbeddingNet_cov, TripletNet, SiameseNet
from config import DATA_CONFIG
import os
import random
import numpy as np
import csv
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


def create_test_loader():
    """
    创建完整的测试数据加载器

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
        # "preprocess": DATA_CONFIG["preprocess"],
        "train_ratio": DATA_CONFIG["train_ratio"],
        "split_seed": DATA_CONFIG["split_seed"],
        "dataset_type": DATA_CONFIG["dataset_type"],
        "normalize_way": DATA_CONFIG["normalize_way"],
    }

    # 创建数据集并设置为验证模式
    test_dataset = IQDataset(**dataset_params, split_mode="val")

    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,  # 增大批量大小以提高效率
        shuffle=False,
        num_workers=0,  # 单进程加载
        pin_memory=False,
    )

    return test_loader


def test_model_performance(embedding_model, classifier_model, test_loader):
    """
    测试模型性能，计算平均推理时间和准确率

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


def test_model_detailed_accuracy(embedding_model, classifier_model, test_loader, num_classes):
    """
    测试模型详细准确率，包括每个类别的准确率、精确率、召回率和F1值，以及总体指标

    参数:
        embedding_model: 嵌入模型
        classifier_model: 分类模型
        test_loader: 测试数据加载器
        num_classes: 类别总数

    返回:
        overall_accuracy: 总体准确率
        class_metrics: 每个类别的指标字典
        confusion_matrix: 混淆矩阵
        macro_precision: 宏平均精确率
        macro_recall: 宏平均召回率
        macro_f1: 宏平均F1值
        micro_precision: 微平均精确率
        micro_recall: 微平均召回率
        micro_f1: 微平均F1值
    """
    # 初始化混淆矩阵
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    with torch.no_grad():
        for data, target in test_loader:
            # 通过嵌入模型和分类模型
            embedding = embedding_model.get_embedding(data)
            output = classifier_model(embedding)
            pred = output.argmax(dim=1)

            # 更新混淆矩阵
            for t, p in zip(target.cpu().numpy(), pred.cpu().numpy()):
                confusion_matrix[t][p] += 1

    # 计算总体准确率
    overall_correct = np.trace(confusion_matrix)
    overall_total = confusion_matrix.sum()
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0

    # 计算每个类别的指标及累计值
    class_metrics = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for i in range(num_classes):
        # 真正例 (True Positives)
        tp = confusion_matrix[i][i]
        # 假正例 (False Positives)
        fp = confusion_matrix[:, i].sum() - tp
        # 假负例 (False Negatives)
        fn = confusion_matrix[i, :].sum() - tp

        # 累计用于微平均计算
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # 准确率 (Accuracy)
        class_total = confusion_matrix[i, :].sum()
        accuracy = tp / class_total if class_total > 0 else 0

        # 精确率 (Precision)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # 召回率 (Recall)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # F1值 (F1-Score)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        class_metrics[i] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    # 计算宏平均指标 (每个类别的算术平均值)
    macro_precision = np.mean([class_metrics[i]['precision'] for i in range(num_classes)])
    macro_recall = np.mean([class_metrics[i]['recall'] for i in range(num_classes)])
    macro_f1 = np.mean([class_metrics[i]['f1'] for i in range(num_classes)])

    # 计算微平均指标 (基于总体TP/FP/FN)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (
                                                                                                      micro_precision + micro_recall) > 0 else 0

    return overall_accuracy, class_metrics, confusion_matrix, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1

def save_metrics_to_csv(overall_accuracy, class_metrics, confusion_matrix, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1, filename="test_metrics.csv"):
    """
    将评估指标结果保存到CSV文件，包括准确率、精确率、召回率和F1值

    参数:
        overall_accuracy: 总体准确率
        class_metrics: 每个类别的指标字典
        confusion_matrix: 混淆矩阵
        macro_precision: 宏平均精确率
        macro_recall: 宏平均召回率
        macro_f1: 宏平均F1值
        micro_precision: 微平均精确率
        micro_recall: 微平均召回率
        micro_f1: 微平均F1值
        filename: CSV文件名
    """
    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # 写入总体指标
        writer.writerow(['总体指标', '值'])
        writer.writerow(['总体准确率', f'{overall_accuracy:.4f}'])
        writer.writerow(['宏平均精确率', f'{macro_precision:.4f}'])
        writer.writerow(['宏平均召回率', f'{macro_recall:.4f}'])
        writer.writerow(['宏平均F1值', f'{macro_f1:.4f}'])
        writer.writerow(['微平均精确率', f'{micro_precision:.4f}'])
        writer.writerow(['微平均召回率', f'{micro_recall:.4f}'])
        writer.writerow(['微平均F1值', f'{micro_f1:.4f}'])
        writer.writerow([])  # 空行分隔


        # 写入每个类别的指标
        writer.writerow(['类别', '准确率', '精确率', '召回率', 'F1值'])
        for class_id, metrics in class_metrics.items():
            writer.writerow([
                f'Class_{class_id}',
                f'{metrics["accuracy"]:.4f}',
                f'{metrics["precision"]:.4f}',
                f'{metrics["recall"]:.4f}',
                f'{metrics["f1"]:.4f}'
            ])
        writer.writerow([])  # 空行分隔

        # 写入混淆矩阵
        writer.writerow(['混淆矩阵', ''] + [f'预测_{i}' for i in range(len(class_metrics))])
        for i, row in enumerate(confusion_matrix):
            writer.writerow([f'实际_{i}'] + [str(x) for x in row])

    print(f"评估指标结果已保存到 {filename}")


def main():
    # 设置随机种子以确保可重复性
    random.seed(42)
    torch.manual_seed(42)

    # TODO 模型路径
    # embedding_model_path = "models/best_model.pth"
    # classifier_model_path = "models/best_model.pth_Classifier.pth"
    embedding_model_path = "models/best_model.pth"
    classifier_model_path = "models/mlp_resnet/best_model.pth_Classifier.pth"

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
    test_loader = create_test_loader()

    print("开始性能测试...")
    avg_time, correct, total = test_model_performance(
        embedding_model, classifier_model, test_loader
    )

    # 打印测试结果
    print("\n测试结果:")
    print(f"平均推理时间: {avg_time:.2f} 毫秒")
    print(f"准确率: {correct}/{total} ({100. * correct / total:.2f}%)")

    print("\n开始详细准确率测试...")
    # 获取类别数量
    num_classes = DATA_CONFIG["class_num"]

    # 测试详细指标
    overall_accuracy, class_metrics, confusion_matrix, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = test_model_detailed_accuracy(
        embedding_model, classifier_model, test_loader, num_classes
    )

    # 打印详细结果
    print(f"\n详细评估指标结果:")
    print(f"准确率: {correct}/{total} ({100. * correct / total:.2f}%)")

    # 新增总体宏平均和微平均指标打印
    print("\n总体评估指标:")
    print(f"宏平均精确率: {correct}/{total} ({100. * correct / total:.2f}%)")
    print(f"宏平均召回率: {macro_recall:.4f}")
    print(f"宏平均F1值:   {macro_f1:.4f}")
    print(f"微平均精确率: {micro_precision:.4f}")
    print(f"微平均召回率: {micro_recall:.4f}")
    print(f"微平均F1值:   {micro_f1:.4f}")

    print("\n各类别指标:")
    print(f"{'类别':<8} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1值':<10}")
    for class_id, metrics in class_metrics.items():
        print(
            f"Class_{class_id:<3} {metrics['accuracy']:.4f}    {metrics['precision']:.4f}    {metrics['recall']:.4f}    {metrics['f1']:.4f}")

    # 保存结果到CSV文件 (更新参数)
    save_metrics_to_csv(overall_accuracy, class_metrics, confusion_matrix, macro_precision, macro_recall, macro_f1,
                        micro_precision, micro_recall, micro_f1, "models/test_metrics_detailed.csv")

if __name__ == "__main__":
    main()
