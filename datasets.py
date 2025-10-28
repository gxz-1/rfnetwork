import numpy as np
import os
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import h5py
from tqdm import tqdm
import copy
from config import DATA_CONFIG, MODULE_CONFIG


class IQDataset(Dataset):
    def __init__(
        self,
        data_dir,
        class_num,
        samples_per_frame=4,
        samples_length=1056 * 75,
        stride=1056 * 15,
        cache_size=20,
        normalize=True,
        train_ratio=0.8,
        split_mode="train",
        split_seed=42,
        dataset_type="single",
        normalize_way="minmax",
    ):
        """
        参数说明：
        - data_dir: 数据目录
        - class_num: 类别数量
        - samples_per_frame: 每帧提取的样本数
        - samples_length: 每个样本长度(1056*15)
        - cache_size: 缓存大小
        - normalize: 是否进行归一化
        - train_ratio: 训练集占比
        - split_mode: 数据集模式："train"训练集, "val"验证集, "all"全部数据
        - split_seed: 随机种子，确保划分的可重复性
        - dataset_type: 数据集组织方式，"single"表示单目录，"recursive"表示递归目录
        """
        super().__init__()
        self.data_dir = data_dir
        self.class_num = class_num
        self.samples_per_frame = samples_per_frame
        self.samples_length = samples_length
        self.stride = stride
        self.cache = {}
        self.cache_size = cache_size
        self.cache_order = []
        self.normalize = normalize
        self.sample_cache = {}
        self.train_ratio = train_ratio
        self.split_mode = split_mode
        self.split_seed = split_seed
        self.dataset_type = dataset_type
        self.normalize_way = normalize_way

        # 根据数据集类型加载文件
        if self.dataset_type == "single":
            # 单目录模式：仅选择前class_num个文件
            self.data_files = sorted(
                [
                    f
                    for f in os.listdir(data_dir)
                    if f.endswith(".dat") or f.endswith(".npy") or f.endswith(".mat")
                ][:class_num]
            )
            self.labels = list(range(len(self.data_files)))
        elif self.dataset_type == "recursive":
            # 递归目录模式：每个子目录作为一个类别
            self.class_dirs = []  # 存储类别目录
            self.data_files = []  # 存储所有文件路径
            self.file_to_class = {}  # 文件到类别的映射
            self.labels = []  # 存储标签

            # 获取所有子目录（类别）
            subdirs = sorted(
                [
                    d
                    for d in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir, d))
                ]
            )
            if class_num > 0:
                subdirs = subdirs[:class_num]  # 限制类别数量

            # 遍历每个类别目录
            for class_idx, subdir in enumerate(subdirs):
                self.class_dirs.append(subdir)
                class_path = os.path.join(data_dir, subdir)

                # 递归获取该类别下的所有文件
                for root, _, files in os.walk(class_path):
                    for file in files:
                        if (
                            file.endswith(".dat")
                            or file.endswith(".npy")
                            or file.endswith(".mat")
                        ):
                            # 存储相对于data_dir的路径
                            rel_path = os.path.relpath(
                                os.path.join(root, file), data_dir
                            )
                            self.data_files.append(rel_path)
                            self.file_to_class[rel_path] = class_idx
                            self.labels.append(class_idx)

        # 初始化帧计数映射
        self.file_frames = {}  # 存储每个文件的帧数
        self.file_offsets = []  # 存储每个文件起始样本索引的偏移量
        self.total_samples = 0  # 总样本数

        # 计算每个文件的帧数和样本偏移量
        self._compute_file_frames()

        # 划分训练集和验证集索引
        self._split_dataset()

        # 如果需要归一化，计算数据统计信息
        base_filename = (
            f"dataset_c{DATA_CONFIG['class_num']}"
            f"_s{DATA_CONFIG['samples_per_frame']}"
            f"_l{DATA_CONFIG['samples_length']}"
            f"_stride{DATA_CONFIG['stride']}"
            f"_r{int(DATA_CONFIG['train_ratio']*100)}"
            f"_seed{DATA_CONFIG['split_seed']}"
            f"_{MODULE_CONFIG['dataset']}"
            f"_{DATA_CONFIG['dataset_type']}"
            f"_norm{DATA_CONFIG['normalize_way']}"
        )
        if self.normalize:
            if os.path.exists(f"DATA/{base_filename}_stats.npy"):
                self.stats = np.load(f"DATA/{base_filename}_stats.npy", allow_pickle=True).item()
                if self.stats["class_num"] != self.class_num:
                    self.stats = self._compute_statistics(self.normalize_way)
            else:
                self.stats = self._compute_statistics(self.normalize_way)

    def _compute_file_frames(self):
        """计算每个文件的帧数，以及样本索引的偏移量"""
        offset = 0
        self.file_offsets.append(offset)  # 第一个文件的偏移量为0

        for file_idx, file_name in tqdm(
            enumerate(self.data_files),
            desc="计算文件帧进度",
            total=len(self.data_files),
        ):
            # 加载数据文件
            data = self._load_file(file_idx)

            # 计算文件中的总样本数
            # 使用步长计算可用的样本数，考虑每个样本的长度
            samples_in_file = max(
                1, (len(data) - self.samples_length) // self.stride + 1
            )
            frames_in_file = (
                samples_in_file // self.samples_per_frame
            )  # 存在向下取整的问题
            self.file_frames[file_idx] = frames_in_file

            # 更新总样本数
            self.total_samples += frames_in_file * self.samples_per_frame

            # 计算下一个文件的偏移量
            if file_idx < len(self.data_files) - 1:
                offset += samples_in_file
                self.file_offsets.append(offset)
        if self.dataset_type == "single":
            print(
                f"数据文件统计: 共{len(self.data_files)}个类别，总样本数{self.total_samples}"
            )
            print(
                "每个类别的帧数:",
                ", ".join(
                    [f"类别{i}: {frames}" for i, frames in self.file_frames.items()]
                ),
            )
        elif self.dataset_type == "recursive":
            print(
                f"数据文件统计: 共{len(self.data_files)}个文件，总样本数{self.total_samples}"
            )
            # 计算每个类别的帧数
            class_frames = {}
            for file_idx, file_name in enumerate(self.data_files):
                class_idx = self.file_to_class[file_name]
                if class_idx not in class_frames:
                    class_frames[class_idx] = 0
                class_frames[class_idx] += self.file_frames[file_idx]
            print(
                "每个类别的帧数：",
                ",".join(
                    [f"类别{key}: {value}" for key, value in class_frames.items()]
                ),
            )

    def _split_dataset(self):
        """
        按类别划分训练集和验证集
        将每个类别的样本按train_ratio比例划分，保存索引以确保可复现性
        """
        # 创建索引文件路径
        base_filename = (
            f"dataset_c{DATA_CONFIG['class_num']}"
            f"_s{DATA_CONFIG['samples_per_frame']}"
            f"_l{DATA_CONFIG['samples_length']}"
            f"_stride{DATA_CONFIG['stride']}"
            f"_r{int(DATA_CONFIG['train_ratio']*100)}"
            f"_seed{DATA_CONFIG['split_seed']}"
            f"_{MODULE_CONFIG['dataset']}"
            f"_{DATA_CONFIG['dataset_type']}"
            f"_norm{DATA_CONFIG['normalize_way']}"
        )
        split_dir = os.path.join("DATA", "splits")
        os.makedirs(split_dir, exist_ok=True)
        split_file = os.path.join(
            split_dir,
            base_filename + f"_{self.split_mode}_split_indices.npy",
        )

        # 如果索引文件存在，直接加载
        if os.path.exists(split_file):
            split_indices = np.load(split_file, allow_pickle=True).item()
            self.train_indices = split_indices["train"]
            self.val_indices = split_indices["val"]
            print(
                f"加载已有的训练集/验证集划分，训练集: {len(self.train_indices)}，验证集: {len(self.val_indices)}"
            )
        else:
            # 为每个类别创建样本索引，然后随机划分
            self.train_indices = []
            self.val_indices = []

            # 设置随机种子以确保可重复性
            random_state = np.random.RandomState(self.split_seed)

            # 对每个文件进行划分
            for class_idx in range(len(self.data_files)):
                # 计算当前文件的样本范围
                samples_in_class = self.file_frames[class_idx] * self.samples_per_frame
                start_idx = self.file_offsets[class_idx]
                end_idx = start_idx + samples_in_class

                # 获取当前文件的所有样本索引
                class_indices = list(range(start_idx, end_idx))

                # 打乱索引
                random_state.shuffle(class_indices)

                # 按比例划分
                train_size = int(len(class_indices) * self.train_ratio)
                class_train_indices = class_indices[:train_size]
                class_val_indices = class_indices[train_size:]

                # 添加到总索引列表
                self.train_indices.extend(class_train_indices)
                self.val_indices.extend(class_val_indices)

            # 保存划分结果
            split_indices = {"train": self.train_indices, "val": self.val_indices}
            np.save(split_file, split_indices)
            print(
                f"创建并保存新的训练集/验证集划分，训练集: {len(self.train_indices)}，验证集: {len(self.val_indices)}"
            )

        # 根据模式选择使用的索引
        if self.split_mode == "train":
            self.indices = self.train_indices
        elif self.split_mode == "val":
            self.indices = self.val_indices
        else:  # "all"模式
            self.indices = list(range(self.total_samples))

    def change_split_mode(self, split_mode):
        """
        修改数据集的模式（训练/验证/全部）

        参数:
            split_mode: str, 可选值为 "train", "val", "all"

        示例:
            dataset.change_split_mode("train")  # 切换到训练模式
            dataset.change_split_mode("val")    # 切换到验证模式
            dataset.change_split_mode("all")    # 使用所有数据
        """
        if split_mode not in ["train", "val", "all"]:
            raise ValueError('split_mode 必须是 "train", "val" 或 "all" 之一')

        # 更新模式
        self.split_mode = split_mode

        # 根据模式选择使用的索引
        if split_mode == "train":
            self.indices = self.train_indices
        elif split_mode == "val":
            self.indices = self.val_indices
        else:  # "all" 模式
            self.indices = list(range(self.total_samples))

        # 清空缓存
        self.cache = {}
        self.cache_order = []
        self.sample_cache = {}

        # 如果是派生类，可能需要更新额外的属性
        if hasattr(self, "class_to_indices"):
            # 更新类别索引映射
            self.class_to_indices = {}
            for class_idx in range(self.num_classes):
                self.class_to_indices[class_idx] = [
                    i
                    for i in self.indices
                    if self._find_file_and_sample(i)[0] == class_idx
                ]

        # 如果是均衡采样的数据集，重置相关计数器
        if hasattr(self, "pair_counters"):
            self.pair_counters = {pair: 0 for pair in self.all_pairs}

        if hasattr(self, "negative_pair_idx"):
            self.negative_pair_idx = 0
            np.random.shuffle(self.negative_pairs)

        if hasattr(self, "enable_stats") and self.enable_stats:
            self.positive_counters = {cls: 0 for cls in range(self.num_classes)}
            self.class_pair_counters = {}

        print(f"数据集模式已切换到: {split_mode}，样本数量: {len(self.indices)}")

    def _compute_statistics(self, normalize_way):
        """
        计算数据集的统计信息（每个时间点的最大最小值）
        使用增量计算方法，减少内存占用
        """
        base_filename = (
            f"dataset_c{DATA_CONFIG['class_num']}"
            f"_s{DATA_CONFIG['samples_per_frame']}"
            f"_l{DATA_CONFIG['samples_length']}"
            f"_stride{DATA_CONFIG['stride']}"
            f"_r{int(DATA_CONFIG['train_ratio']*100)}"
            f"_seed{DATA_CONFIG['split_seed']}"
            f"_{MODULE_CONFIG['dataset']}"
            f"_{DATA_CONFIG['dataset_type']}"
            f"_norm{DATA_CONFIG['normalize_way']}"
        )
        if normalize_way == "minmax":
            print("计算数据集统计信息...")

            # 初始化统计量，为每个时间点创建最大最小值数组
            real_min = np.full(self.samples_length, float("inf"))
            real_max = np.full(self.samples_length, float("-inf"))
            imag_min = np.full(self.samples_length, float("inf"))
            imag_max = np.full(self.samples_length, float("-inf"))

            # 对每个文件进行处理
            for file_idx, _ in tqdm(
                enumerate(self.data_files),
                desc="文件处理进度",
                total=len(self.data_files),
            ):
                data = self._load_file(file_idx)
                frames_in_file = self.file_frames[file_idx]

                # 遍历该文件中的所有有效样本
                for frame_idx in range(frames_in_file):
                    # 计算当前帧的起始位置，每帧有self.samples_per_frame个样本，每个样本长度为self.samples_length，每个样本之间间隔self.stride
                    frame_start = frame_idx * self.samples_per_frame * self.stride

                    for sample_idx in range(self.samples_per_frame):
                        # 计算样本位置
                        sample_start = frame_start + sample_idx * self.stride
                        sample_end = sample_start + self.samples_length

                        # 确保不超出数据范围
                        if sample_end <= len(data):
                            # 提取样本
                            sample = data[sample_start:sample_end]

                            # 更新每个时间点的最大最小值
                            real_min = np.minimum(real_min, sample.real)
                            real_max = np.maximum(real_max, sample.real)
                            imag_min = np.minimum(imag_min, sample.imag)
                            imag_max = np.maximum(imag_max, sample.imag)

            stats = {
                "class_num": self.class_num,
                "real_min": real_min.tolist(),  # 转换为列表以便保存,只显示前10个
                "real_max": real_max.tolist(),
                "imag_min": imag_min.tolist(),
                "imag_max": imag_max.tolist(),
            }

            print("\n数据统计信息：")
            print(
                f"实部 - 最小值范围: [{np.min(real_min):.6f}, {np.max(real_min):.6f}]"
            )
            print(
                f"实部 - 最大值范围: [{np.min(real_max):.6f}, {np.max(real_max):.6f}]"
            )
            print(
                f"虚部 - 最小值范围: [{np.min(imag_min):.6f}, {np.max(imag_min):.6f}]"
            )
            print(
                f"虚部 - 最大值范围: [{np.min(imag_max):.6f}, {np.max(imag_max):.6f}]"
            )

            # 保存统计信息到文件
            np.save(f"DATA/{base_filename}_stats.npy", stats)

            return stats
        elif normalize_way == "zscore":
            print("计算数据集统计信息...")

            # 初始化计数器和累加器
            count = 0
            real_sum = 0
            imag_sum = 0
            real_square_sum = 0
            imag_square_sum = 0
            real_min = float("inf")
            real_max = float("-inf")
            imag_min = float("inf")
            imag_max = float("-inf")

            # 对每个文件进行处理
            for file_idx, _ in tqdm(
                enumerate(self.data_files),
                desc="文件处理进度",
                total=len(self.data_files),
            ):
                data = self._load_file(file_idx)
                frames_in_file = self.file_frames[file_idx]

                # 遍历该文件中的所有有效样本
                for frame_idx in range(frames_in_file):
                    frame_start = frame_idx * self.samples_per_frame * self.stride

                    for sample_idx in range(self.samples_per_frame):
                        # 计算样本位置
                        sample_start = frame_start + sample_idx * self.samples_length
                        sample_end = sample_start + self.samples_length

                        # 确保不超出数据范围
                        if sample_end <= len(data):
                            # 提取样本
                            sample = data[sample_start:sample_end]

                            # 增量更新统计量
                            real_part = sample.real
                            imag_part = sample.imag

                            real_sum += np.sum(real_part)
                            imag_sum += np.sum(imag_part)
                            real_square_sum += np.sum(np.square(real_part))
                            imag_square_sum += np.sum(np.square(imag_part))
                            count += len(sample)

                            # 更新最大最小值
                            real_min = min(real_min, np.min(real_part))
                            real_max = max(real_max, np.max(real_part))
                            imag_min = min(imag_min, np.min(imag_part))
                            imag_max = max(imag_max, np.max(imag_part))

            # 计算最终的统计量
            real_mean = real_sum / count
            imag_mean = imag_sum / count
            real_std = np.sqrt(real_square_sum / count - real_mean**2) + 1e-6
            imag_std = np.sqrt(imag_square_sum / count - imag_mean**2) + 1e-6

            stats = {
                "class_num": self.class_num,
                "real_mean": float(real_mean),
                "real_std": float(real_std),
                "imag_mean": float(imag_mean),
                "imag_std": float(imag_std),
                "real_min": float(real_min),
                "real_max": float(real_max),
                "imag_min": float(imag_min),
                "imag_max": float(imag_max),
            }

            print("\n数据统计信息：")
            print(
                f"实部 - 均值: {stats['real_mean']:.6f}, 标准差: {stats['real_std']:.6f}"
            )
            print(
                f"实部 - 最小值: {stats['real_min']:.6f}, 最大值: {stats['real_max']:.6f}"
            )
            print(
                f"虚部 - 均值: {stats['imag_mean']:.6f}, 标准差: {stats['imag_std']:.6f}"
            )
            print(
                f"虚部 - 最小值: {stats['imag_min']:.6f}, 最大值: {stats['imag_max']:.6f}"
            )
            # 保存统计信息到文件
            np.save(f"DATA/{base_filename}_stats.npy", stats)

            return stats

    def _find_file_and_sample(self, idx):
        """查找绝对索引对应的文件和样本位置"""
        # 找到文件索引
        file_idx = 0
        for i, offset in enumerate(self.file_offsets):
            if i < len(self.file_offsets) - 1 and idx >= self.file_offsets[i + 1]:
                continue
            file_idx = i
            break

        # 计算文件内相对偏移
        relative_idx = idx - self.file_offsets[file_idx]

        # 计算帧索引和样本索引
        frame_idx = relative_idx // self.samples_per_frame
        sample_idx = relative_idx % self.samples_per_frame

        return file_idx, frame_idx, sample_idx

    def __len__(self):
        # 返回当前模式下的样本数量
        return len(self.indices)

    def __getitem__(self, idx):
        # 根据当前模式获取实际索引
        actual_idx = self.indices[idx]

        # 检查样本缓存
        if actual_idx in self.sample_cache:
            return self.sample_cache[actual_idx]

        # 计算索引
        file_idx, frame_idx, sample_idx = self._find_file_and_sample(actual_idx)

        # 加载数据文件
        data = self._load_file(file_idx)

        # 计算样本位置，考虑步长
        # 计算当前样本在整个文件中的绝对位置
        absolute_sample_idx = frame_idx * self.samples_per_frame + sample_idx
        sample_start = absolute_sample_idx * self.stride
        sample_end = sample_start + self.samples_length

        # 确保索引在有效范围内
        if sample_end > len(data):
            # 超出范围，创建零填充样本
            sample = np.zeros(self.samples_length, dtype=np.complex64)
        else:
            # 提取样本
            sample = data[sample_start:sample_end]

        # 归一化
        if self.normalize:
            sample = self.normalize_sample(sample)

        # 转换为张量
        sample_tensor = torch.FloatTensor(np.stack([sample.real, sample.imag]))

        # 获取标签
        if self.dataset_type == "single":
            label = file_idx
        else:
            label = self.file_to_class[self.data_files[file_idx]]

        # 缓存样本
        if len(self.sample_cache) < 1000:  # 只缓存1000个样本
            self.sample_cache[actual_idx] = (sample_tensor, label)

        return sample_tensor, label

    def normalize_sample(self, sample):
        """
        对样本进行MinMax归一化
        使用总体最大最小值对实部和虚部分别进行归一化
        """
        if not self.normalize:
            return sample
        if self.normalize_way == "minmax":
            # 将统计量转换为numpy数组
            real_min = np.array(self.stats["real_min"])
            real_max = np.array(self.stats["real_max"])
            imag_min = np.array(self.stats["imag_min"])
            imag_max = np.array(self.stats["imag_max"])

            # 添加epsilon防止除零错误
            epsilon = 1e-8
            real_range = np.where((real_max - real_min) < epsilon, epsilon, real_max - real_min)
            imag_range = np.where((imag_max - imag_min) < epsilon, epsilon, imag_max - imag_min)

            real_norm = (sample.real - real_min) / real_range
            imag_norm = (sample.imag - imag_min) / imag_range
            return real_norm + 1j * imag_norm
        elif self.normalize_way == "zscore":
            # 将统计量转换为numpy数组
            # 使用Z-score归一化，从字典中正确获取统计值
            real_norm = (sample.real - self.stats["real_mean"]) / self.stats["real_std"]
            imag_norm = (sample.imag - self.stats["imag_mean"]) / self.stats["imag_std"]

            return real_norm + 1j * imag_norm

    def _load_file(self, file_idx):
        """加载文件，优先使用预处理的二进制文件"""
        if file_idx in self.cache:
            self.cache_order.remove(file_idx)
            self.cache_order.append(file_idx)
            return self.cache[file_idx]

        file_name = self.data_files[file_idx]
        file_path = os.path.join(self.data_dir, file_name)
        if file_path.endswith(".mat"):
            try:
                mat_data = sio.loadmat(file_path)
                data = mat_data["data"]
            except NotImplementedError:
                # 如果是MATLAB v7.3格式，使用h5py读取
                with h5py.File(file_path, "r") as f:
                    # 读取数据
                    data = np.array(f["iqdata"])

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
                    data = np.array(iq_data, dtype=np.complex64).flatten()
        elif file_path.endswith(".txt"):
            with open(file_path, "r") as f:
                lines = f.readlines()
                data = np.array(
                    [complex(line.strip().replace("i", "j")) for line in lines],
                    dtype=np.complex64,
                )
        elif file_path.endswith(".npy"):
            data = np.load(file_path)

        # 管理缓存
        if len(self.cache) >= self.cache_size:
            oldest_idx = self.cache_order.pop(0)
            del self.cache[oldest_idx]

        self.cache[file_idx] = data
        self.cache_order.append(file_idx)
        return data

    def __deepcopy__(self, memo):
        """
        创建数据集的深度复制
        """
        # 创建一个新的实例，但不调用__init__
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # 复制所有属性
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))

        return result


class SiameseIQDataset_random(IQDataset):
    def __getitem__(self, idx):
        # 获取基础样本
        sample1, label1 = super().__getitem__(idx)

        # 随机选择正样本或负样本
        if np.random.random() < 0.5:
            # 正样本：从同一文件随机选择另一个样本
            same_file_idx = label1  # label1就是file_idx

            # 获取当前类的所有训练/验证集索引
            class_indices = [
                i
                for i in self.indices
                if self._find_file_and_sample(i)[0] == same_file_idx
            ]

            if len(class_indices) <= 1:
                # 如果当前类只有一个样本，使用它自己
                sample2 = sample1
            else:
                # 随机选择同类的另一个样本（不包括当前样本）
                actual_idx = self.indices[idx]
                available_indices = [i for i in class_indices if i != actual_idx]
                if not available_indices:
                    sample2 = sample1
                else:
                    sample2_actual_idx = np.random.choice(available_indices)
                    sample2_idx = self.indices.index(sample2_actual_idx)
                    sample2, _ = IQDataset.__getitem__(self, sample2_idx)

            target = 1
        else:
            # 负样本：从不同文件随机选择样本
            current_file_idx = label1

            # 获取不同类的所有训练/验证集索引
            different_class_indices = [
                i
                for i in self.indices
                if self._find_file_and_sample(i)[0] != current_file_idx
            ]

            if not different_class_indices:
                # 如果没有其他类，使用当前样本
                sample2 = sample1
                target = 1  # 强制为正样本
            else:
                # 随机选择不同类的样本
                sample2_actual_idx = np.random.choice(different_class_indices)
                sample2_idx = self.indices.index(sample2_actual_idx)
                sample2, _ = IQDataset.__getitem__(self, sample2_idx)
                target = 0

        return (sample1, sample2), target

    def __deepcopy__(self, memo):
        """
        创建数据集的深度复制
        """
        # 调用父类的深度复制方法
        result = super().__deepcopy__(memo)
        return result


class SiameseIQDataset_balanced_1(IQDataset):
    """
    均衡的孪生网络数据集，确保不同类别对之间的负样本数量均衡
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = len(self.data_files)

        # 为每个类别创建样本索引池
        self.class_to_indices = {}
        for class_idx in range(self.num_classes):
            self.class_to_indices[class_idx] = [
                i for i in self.indices if self._find_file_and_sample(i)[0] == class_idx
            ]
            np.random.shuffle(self.class_to_indices[class_idx])

        # 预生成所有可能的类别对
        self.all_pairs = []
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                self.all_pairs.append((i, j))

        # 为每个类别对维护一个计数器
        self.pair_counters = {pair: 0 for pair in self.all_pairs}

    def __getitem__(self, idx):
        sample1, label1 = super().__getitem__(idx)
        current_class = label1

        if np.random.random() < 0.5:
            # 正样本：从同一类别随机选择一个样本
            class_indices = self.class_to_indices[current_class]

            if len(class_indices) <= 1:
                # 如果当前类只有一个样本，使用它自己
                sample2 = sample1
            else:
                # 随机选择同类的另一个样本（不包括当前样本）
                actual_idx = self.indices[idx]
                available_indices = [i for i in class_indices if i != actual_idx]
                if not available_indices:
                    sample2 = sample1
                else:
                    sample2_actual_idx = np.random.choice(available_indices)
                    # 查找索引在当前模式下的位置
                    if sample2_actual_idx in self.indices:
                        sample2_idx = self.indices.index(sample2_actual_idx)
                        sample2, _ = IQDataset.__getitem__(self, sample2_idx)
                    else:
                        # 如果不在当前模式的索引中，直接获取样本
                        sample2 = self._get_sample_by_absolute_idx(sample2_actual_idx)

            target = 1
        else:
            # 负样本：选择使用次数最少的类别对
            available_pairs = [pair for pair in self.all_pairs if current_class in pair]
            if not available_pairs:
                available_pairs = [
                    (current_class, other)
                    for other in range(self.num_classes)
                    if other != current_class
                ]

            # 选择使用次数最少的类别对
            min_count = min(self.pair_counters[pair] for pair in available_pairs)
            candidate_pairs = [
                pair
                for pair in available_pairs
                if self.pair_counters[pair] == min_count
            ]
            selected_pair = candidate_pairs[np.random.randint(len(candidate_pairs))]

            # 更新计数器
            self.pair_counters[selected_pair] += 1

            # 确定另一个类别
            other_class = (
                selected_pair[0]
                if selected_pair[0] != current_class
                else selected_pair[1]
            )

            # 从另一个类别选择样本
            other_class_indices = self.class_to_indices[other_class]
            # 随机选择不同类的样本
            sample2_actual_idx = np.random.choice(other_class_indices)
            # 查找索引在当前模式下的位置
            if sample2_actual_idx in self.indices:
                sample2_idx = self.indices.index(sample2_actual_idx)
                sample2, _ = IQDataset.__getitem__(self, sample2_idx)
            else:
                # 如果不在当前模式的索引中，直接获取样本
                sample2 = self._get_sample_by_absolute_idx(sample2_actual_idx)
            target = 0

        return (sample1, sample2), target

    def _get_sample_by_absolute_idx(self, absolute_idx):
        """通过绝对索引获取样本"""
        # 直接调用父类方法
        return super()._get_sample_by_absolute_idx(absolute_idx)

    def __deepcopy__(self, memo):
        """
        创建数据集的深度复制
        """
        # 调用父类的深度复制方法
        result = super().__deepcopy__(memo)

        # 复制特有的属性
        result.pair_counters = copy.deepcopy(self.pair_counters, memo)
        result.all_pairs = copy.deepcopy(self.all_pairs, memo)
        result.class_to_indices = copy.deepcopy(self.class_to_indices, memo)

        return result


class SiameseIQDataset_balanced_2(IQDataset):
    """
    均衡的孪生网络数据集，确保不同类别对之间的负样本数量均衡
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = len(self.data_files)

        # 预生成负样本类别对，但使用循环方式访问
        self.negative_pairs = []
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                self.negative_pairs.append((i, j))
        self.negative_pair_idx = 0
        np.random.shuffle(self.negative_pairs)

        # 为每个类别创建样本索引词典
        self.class_to_indices = {}
        for class_idx in range(self.num_classes):
            self.class_to_indices[class_idx] = [
                i for i in self.indices if self._find_file_and_sample(i)[0] == class_idx
            ]

        # 简化计数器，仅在需要时统计
        self.enable_stats = False
        if self.enable_stats:
            self.positive_counters = {cls: 0 for cls in range(self.num_classes)}
            self.class_pair_counters = {}

    def __getitem__(self, idx):
        sample1, label1 = super().__getitem__(idx)
        current_class = label1

        if np.random.random() < 0.5:
            # 正样本：从同一类别随机选择一个样本
            class_indices = self.class_to_indices[current_class]

            if len(class_indices) <= 1:
                # 如果当前类只有一个样本，使用它自己
                sample2 = sample1
            else:
                # 随机选择同类的另一个样本（不包括当前样本）
                actual_idx = self.indices[idx]
                available_indices = [i for i in class_indices if i != actual_idx]
                if not available_indices:
                    sample2 = sample1
                else:
                    sample2_actual_idx = np.random.choice(available_indices)
                    # 查找索引在当前模式下的位置
                    if sample2_actual_idx in self.indices:
                        sample2_idx = self.indices.index(sample2_actual_idx)
                        sample2, _ = IQDataset.__getitem__(self, sample2_idx)
                    else:
                        # 如果不在当前模式的索引中，直接获取样本
                        sample2 = self._get_sample_by_absolute_idx(sample2_actual_idx)

            target = 1

            # 仅在需要时更新统计
            if self.enable_stats:
                self.positive_counters[current_class] += 1
        else:
            # 负样本：保留循环访问类别对的方式
            if self.negative_pair_idx >= len(self.negative_pairs):
                self.negative_pair_idx = 0
                np.random.shuffle(self.negative_pairs)

            class_i, class_j = self.negative_pairs[self.negative_pair_idx]
            self.negative_pair_idx += 1

            # 确保当前类别在类别对中
            if current_class != class_i and current_class != class_j:
                replace_idx = np.random.randint(0, 2)
                if replace_idx == 0:
                    class_i = current_class
                else:
                    class_j = current_class

            other_class = class_j if current_class == class_i else class_i

            # 从另一个类别选择样本
            other_class_indices = self.class_to_indices[other_class]
            if not other_class_indices:
                # 如果没有其他类样本，使用当前样本
                sample2 = sample1
                target = 1  # 强制为正样本
            else:
                # 随机选择不同类的样本
                sample2_actual_idx = np.random.choice(other_class_indices)
                # 查找索引在当前模式下的位置
                if sample2_actual_idx in self.indices:
                    sample2_idx = self.indices.index(sample2_actual_idx)
                    sample2, _ = IQDataset.__getitem__(self, sample2_idx)
                else:
                    # 如果不在当前模式的索引中，直接获取样本
                    sample2 = self._get_sample_by_absolute_idx(sample2_actual_idx)
                target = 0

            # 仅在需要时更新统计
            if self.enable_stats:
                pair = tuple(sorted([current_class, other_class]))
                self.class_pair_counters[pair] = (
                    self.class_pair_counters.get(pair, 0) + 1
                )

        return (sample1, sample2), target

    def _get_sample_by_absolute_idx(self, absolute_idx):
        """通过绝对索引获取样本"""
        # 直接调用父类方法
        return super()._get_sample_by_absolute_idx(absolute_idx)

    def __deepcopy__(self, memo):
        """
        创建数据集的深度复制
        """
        # 调用父类的深度复制方法
        result = super().__deepcopy__(memo)

        # 复制特有的属性
        result.negative_pairs = copy.deepcopy(self.negative_pairs, memo)
        result.negative_pair_idx = self.negative_pair_idx
        result.class_to_indices = copy.deepcopy(self.class_to_indices, memo)

        if hasattr(self, "enable_stats") and self.enable_stats:
            result.positive_counters = copy.deepcopy(self.positive_counters, memo)
            result.class_pair_counters = copy.deepcopy(self.class_pair_counters, memo)

        return result


class TripletIQDataset(IQDataset):
    """
    三元组数据集，用于TripletLoss训练
    返回(anchor, positive, negative)三个样本
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = len(self.data_files)

        # 为每个类别创建样本索引词典
        self.class_to_indices = {}
        for class_idx in range(self.num_classes):
            self.class_to_indices[class_idx] = [
                i for i in self.indices if self._find_file_and_sample(i)[0] == class_idx
            ]
            np.random.shuffle(self.class_to_indices[class_idx])

        # 预生成所有可能的类别对
        self.all_pairs = []
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                self.all_pairs.append((i, j))

        # 为每个类别对维护一个计数器
        self.pair_counters = {pair: 0 for pair in self.all_pairs}

    def __getitem__(self, idx):
        # 获取anchor样本
        anchor, anchor_label = super().__getitem__(idx)
        current_class = anchor_label

        # 获取positive样本（同一类别）
        class_indices = self.class_to_indices[current_class]

        if len(class_indices) <= 1:
            # 如果当前类只有一个样本，使用它自己
            positive = anchor
        else:
            # 随机选择同类的另一个样本（不包括当前样本）
            actual_idx = self.indices[idx]
            available_indices = [i for i in class_indices if i != actual_idx]
            if not available_indices:
                positive = anchor
            else:
                pos_actual_idx = np.random.choice(available_indices)
                # 查找索引在当前模式下的位置
                if pos_actual_idx in self.indices:
                    pos_idx = self.indices.index(pos_actual_idx)
                    positive, _ = IQDataset.__getitem__(self, pos_idx)
                else:
                    # 如果不在当前模式的索引中，直接获取样本
                    positive = self._get_sample_by_absolute_idx(pos_actual_idx)

        # 获取negative样本（不同类别）
        # 选择使用次数最少的类别对
        available_pairs = [pair for pair in self.all_pairs if current_class in pair]
        if not available_pairs:
            available_pairs = [
                (current_class, other)
                for other in range(self.num_classes)
                if other != current_class
            ]
        # 选择使用次数最少的类别对
        min_count = min(self.pair_counters[pair] for pair in available_pairs)
        candidate_pairs = [
            pair for pair in available_pairs if self.pair_counters[pair] == min_count
        ]
        selected_pair = candidate_pairs[np.random.randint(len(candidate_pairs))]

        # 更新计数器
        self.pair_counters[selected_pair] += 1

        # 确定另一个类别
        other_class = (
            selected_pair[0] if selected_pair[0] != current_class else selected_pair[1]
        )

        # 从另一个类别选择样本
        other_class_indices = self.class_to_indices[other_class]
        # 随机选择不同类的样本
        neg_actual_idx = np.random.choice(other_class_indices)
        # 查找索引在当前模式下的位置
        if neg_actual_idx in self.indices:
            neg_idx = self.indices.index(neg_actual_idx)
            negative, _ = IQDataset.__getitem__(self, neg_idx)
        else:
            # 如果不在当前模式的索引中，直接获取样本
            negative = self._get_sample_by_absolute_idx(neg_actual_idx)

        return (anchor, positive, negative), []  # 返回空标签，因为TripletLoss不需要标签

    def _get_sample_by_absolute_idx(self, absolute_idx):
        """通过绝对索引获取样本"""
        # 直接调用父类方法
        return super()._get_sample_by_absolute_idx(absolute_idx)

    def __deepcopy__(self, memo):
        """
        创建数据集的深度复制
        """
        # 调用父类的深度复制方法
        result = super().__deepcopy__(memo)

        # 复制特有的属性
        result.all_pairs = copy.deepcopy(self.all_pairs, memo)
        result.pair_counters = copy.deepcopy(self.pair_counters, memo)
        result.class_to_indices = copy.deepcopy(self.class_to_indices, memo)

        return result
