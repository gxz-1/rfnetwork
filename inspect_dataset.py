import numpy as np

def inspect_dat_file(file_path, num_samples=10):
    try:
        # 读取原始二进制数据
        raw_data = np.fromfile(file_path, dtype=np.int32)
        print(f"文件: {file_path}")
        print(f"总数据点数: {len(raw_data)}")
        print(f"前{num_samples}个数据点:")
        print(raw_data[:num_samples])
        
        # 检查数据范围
        print("\n数据范围检查:")
        print(f"最小值: {raw_data.min()}")
        print(f"最大值: {raw_data.max()}")
        print(f"建议范围: [-32768, 32767]")
        
        # 检查异常值
        out_of_range = np.sum((raw_data < -32768) | (raw_data > 32767))
        print(f"\n异常值数量: {out_of_range}")
        
    except Exception as e:
        print(f"读取文件失败: {str(e)}")


def inspect_npy_file(file_path, num_samples=10):
    try:
        # 加载npy文件
        data = np.load(file_path)
        print(f"文件: {file_path}")
        print(f"数据类型: {data.dtype}")
        print(f"数组形状: {data.shape}")

        # 检查是否为复数数据
        if np.iscomplexobj(data):
            print("\n复数数据统计:")
            print(f"实部范围: [{data.real.min():.4f}, {data.real.max():.4f}]")
            print(f"虚部范围: [{data.imag.min():.4f}, {data.imag.max():.4f}]")
            print(f"NaN数量: {np.isnan(data).sum()}")
            print(f"Inf数量: {np.isinf(data).sum()}")

            # 显示样本数据
            print(f"\n前{num_samples}个复数样本:")
            for i in range(num_samples):
                print(f"{data[i]:+.4f}{data[i].imag:+.4f}j")
        else:
            print("\n实数数据统计:")
            print(f"数值范围: [{data.min():.4f}, {data.max():.4f}]")
            print(f"前{num_samples}个样本:")
            print(data[:num_samples])

    except Exception as e:
        print(f"读取文件失败: {str(e)}")


# 使用示例
inspect_npy_file("DATA/preprocessed/1.6663G.npy")
# 使用示例
# inspect_dat_file("DATA/new40classrawdata/中星10/1.6680G.dat")
