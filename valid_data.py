# -*- coding: utf-8 -*-
"""
@Time ： 2025/10/28 18:17
@Auth ： 高夕茁
@File ：valid_data.py
"""
from datasets import IQDataset

# 初始化数据集（参数需与config.py匹配）
try:
    dataset = IQDataset(
        data_dir="DATA/preprocessed",  # 预处理后的目录
        class_num=40,  # 类别数量
        dataset_type="single",  # 递归目录模式（子目录为类别）
        samples_length=1056 * 75,  # 样本长度（需与process.py中的frame_length一致）
        normalize=True,  # 先禁用归一化，简化验证
        split_mode="all"  # 加载所有样本
    )

    print(f"✅ 数据集初始化成功！总样本数: {len(dataset)}")
    print(f"类别数量: {len(dataset.data_files)}")

    # 加载一个样本验证
    sample_tensor, label = dataset[0]
    print(f"样本形状: {sample_tensor.shape} (应为 [2, samples_length])")
    print(f"样本标签: {label} (应为0~39之间的整数)")
    print("✅ 样本加载成功！")

except Exception as e:
    print(f"❌ 数据集加载失败: {e}")