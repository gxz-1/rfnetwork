"""
超参数配置文件
包含训练、模型架构、数据加载和损失函数的所有可配置参数
"""

# 测试模型路径
# TEST_MODEL_PATH = "models/best_model.pth"
TEST_MODEL_PATH = "random/checkpoint_epoch_15.pth"

# 检查点路径
CHECKPOINT_PATH = "models/checkpoint_epoch_10.pth"

# 数据相关参数
DATA_CONFIG = {
    # 数据目录
    "data_dir": "DATA/preprocessed",
    "class_num": 40,  # 类别数量
    # 数据集参数
    "samples_per_frame": 4,  # 每帧提取的样本数
    "frame_length": 1024 * 140,  # 每帧总长度
    "samples_length": 1024 * 35,  # 每个样本长度
    "stride": 1024 * 5,  # 滑动窗口步长，相对于样本进行滑动
    "cache_size": 20,  # 文件缓存大小
    "normalize": True,  # 是否归一化样本
    "normalize_way": "minmax",  # 归一化方式,minmax: 最小-最大归一化, zscore: Z-Score标准化
    # 数据加载器参数
    "batch_size": 32,  # 批次大小
    "num_workers": 0,  # 数据加载线程数
    "pin_memory": True,  # 是否将数据加载到CUDA固定内存
    "prefetch_factor": 0,  # 预取因子
    "shuffle": True,  # 是否打乱训练数据
    # 数据集划分
    "train_ratio": 0.8,  # 训练集占总数据集的比例
    "split_seed": 42,  # 随机种子，确保划分的可重复性
    "dataset_type": "single",  # 数据集源组织格式, single: 单目录, recursive: 递归目录
}

# 训练相关参数
TRAIN_CONFIG = {
    "n_epochs": 20,  # 训练轮数
    "learning_rate": 0.0001,  # 学习率
    "scheduler_step_size": 8,  # 学习率调度器步长
    "scheduler_gamma": 0.1,  # 学习率衰减因子
    "log_interval": 100,  # 日志打印间隔
    "cuda": True,  # 是否使用CUDA
    "cudnn_benchmark": True,  # 是否启用CUDA性能优化
    "cudnn_deterministic": True,  # 是否启用CUDA确定性计算
    # 检查点保存
    "checkpoint_dir": "models/",  # 检查点保存目录
    "save_best": True,  # 是否保存最佳模型
    "save_interval": 4,  # 定期保存检查点的间隔
}

# 损失函数参数
LOSS_CONFIG = {
    "contrastive_margin": 1.0,  # 对比损失间隔
    "triplet_margin": 1.0,  # 三元组损失间隔
}

# 可视化参数
VISUALIZATION_CONFIG = {
    "tsne_perplexity": 30,  # t-SNE困惑度参数
    "tsne_n_iter": 1000,  # t-SNE迭代次数
    "figsize": (10, 8),  # 图像大小
}

# 模块参数
MODULE_CONFIG = {
    "embedding_net": "res",  # 嵌入网络类型, cov: 卷积网络, res: 残差网络
    "dataset": "random",  # 数据集类型, random: 随机数据集, balanced_1: 均衡数据集1, balanced_2: 均衡数据集2, triplet: 三元组数据集, simple: 简单数据集
    "optimizer": "adam",  # 优化器类型, adam: Adam优化器, sgd: SGD优化器
    "scheduler": "step",  # 学习率调度器类型, step: 步进学习率调度器, cosine: 余弦退火学习率调度器
    "classifier": "mlp_resnet",  # 分类器类型, fc: 全连接分类器, mlp_resnet: MLPResNet分类器
    "status": 1,  # 状态, 1: 测试，2: 训练，3：继续训练，4: 验证数据集
}
