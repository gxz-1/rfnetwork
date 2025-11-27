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
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def load_models(embedding_model_path, classifier_model_path):
    # ... existing code ...
    pass


def create_test_loader():
    # ... existing code ...
    pass


def test_model_performance(embedding_model, classifier_model, test_loader):
    # ... existing code ...
    pass


def test_model_detailed_accuracy(embedding_model, classifier_model, test_loader, num_classes):
    # ... existing code ...
    pass


def calculate_metrics(embedding_model, classifier_model, test_loader, num_classes):
    """
    计算模型的详细评估指标，包括准确率、精确率、召回率和F1值

    参数:
        embedding_model: 嵌入模型
        classifier_model: 分类模型
        test_loader: 测试数据加载器
        num_classes: 类别总数

    返回:
        metrics: 包含所有评估指标的字典
    """
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            # 通过嵌入模型
            embedding = embedding_model.get_embedding(data)

            # 通过分类模型
            output = classifier_model(embedding)

            # 计算预测结果
            pred = output.argmax(dim=1)

            # 收集预测结果和真实标签
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.numpy())

    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 计算总体准确率
    overall_accuracy = np.mean(all_preds == all_targets)

    # 计算每个类别的精确率、召回率和F1值
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, average=None, labels=range(num_classes)
    )

    # 计算宏平均和微平均
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro'
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='micro'
    )

    # 计算每个类别的准确率
    class_accuracies = {}
    for i in range(num_classes):
        class_mask = (all_targets == i)
        if np.sum(class_mask) > 0:
            class_accuracies[i] = np.mean(all_preds[class_mask] == i)
        else:
            class_accuracies[i] = 0.0

    # 构建返回的指标字典
    metrics = {
        'overall_accuracy': overall_accuracy,
        'class_accuracies': class_accuracies,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1
    }

    return metrics


def save_metrics_to_csv(metrics, filename="test_metrics.csv"):
    """
    将评估指标保存到CSV文件

    参数:
        metrics: 包含所有评估指标的字典
        filename: CSV文件名
    """
    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # 写入表头
        writer.writerow(['Metric', 'Value'])

        # 写入总体指标
        writer.writerow(['Overall Accuracy', f'{metrics["overall_accuracy"]:.4f}'])
        writer.writerow(['Macro Precision', f'{metrics["macro_precision"]:.4f}'])
        writer.writerow(['Macro Recall', f'{metrics["macro_recall"]:.4f}'])
        writer.writerow(['Macro F1-Score', f'{metrics["macro_f1"]:.4f}'])
        writer.writerow(['Micro Precision', f'{metrics["micro_precision"]:.4f}'])
        writer.writerow(['Micro Recall', f'{metrics["micro_recall"]:.4f}'])
        writer.writerow(['Micro F1-Score', f'{metrics["micro_f1"]:.4f}'])

        # 写入每个类别的指标
        writer.writerow([])  # 空行分隔
        writer.writerow(['Class ID', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Support'])

        num_classes = len(metrics['class_accuracies'])
        for i in range(num_classes):
            writer.writerow([
                f'Class_{i}',
                f'{metrics["class_accuracies"][i]:.4f}',
                f'{metrics["precision"][i]:.4f}' if i < len(metrics["precision"]) else 'N/A',
                f'{metrics["recall"][i]:.4f}' if i < len(metrics["recall"]) else 'N/A',
                f'{metrics["f1"][i]:.4f}' if i < len(metrics["f1"]) else 'N/A',
                f'{metrics["support"][i]}' if i < len(metrics["support"]) else 'N/A'
            ])

    print(f"评估指标已保存到 {filename}")


def main():
    # 设置随机种子以确保可重复性
    random.seed(42)
    torch.manual_seed(42)

    # TODO 模型路径
    embedding_model_path = "models/best_model.pth"
    classifier_model_path = "models/best_model.pth_Classifier.pth"
    # embedding_model_path = "models/checkpoint_epoch_25.pth"
    # classifier_model_path = "models/checkpoint_epoch_25.pth_Classifier.pth"

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

    print("\n开始详细指标测试...")
    # 获取类别数量
    num_classes = DATA_CONFIG["class_num"]

    # 计算详细评估指标
    metrics = calculate_metrics(
        embedding_model, classifier_model, test_loader, num_classes
    )

    # 打印详细结果
    print(f"\n详细评估指标结果:")
    print(f"总体准确率: {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy'] * 100:.2f}%)")
    print(f"宏平均精确率: {metrics['macro_precision']:.4f}")
    print(f"宏平均召回率: {metrics['macro_recall']:.4f}")
    print(f"宏平均F1值: {metrics['macro_f1']:.4f}")
    print(f"微平均精确率: {metrics['micro_precision']:.4f}")
    print(f"微平均召回率: {metrics['micro_recall']:.4f}")
    print(f"微平均F1值: {metrics['micro_f1']:.4f}")

    print("\n各类别指标:")
    for i in range(num_classes):
        precision = metrics['precision'][i] if i < len(metrics['precision']) else 0.0
        recall = metrics['recall'][i] if i < len(metrics['recall']) else 0.0
        f1 = metrics['f1'][i] if i < len(metrics['f1']) else 0.0
        support = metrics['support'][i] if i < len(metrics['support']) else 0
        print(f"类别 {i}: 准确率={metrics['class_accuracies'][i]:.4f}, "
              f"精确率={precision:.4f}, 召回率={recall:.4f}, F1={f1:.4f}, 支持数={support}")

    # 保存结果到CSV文件
    save_metrics_to_csv(metrics, "models/test_metrics_detailed.csv")


if __name__ == "__main__":
    main()