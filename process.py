import os
import numpy as np

def convert_to_IQ(data_file_path,save_path, frame_length=1056 * 75, num_frames=50):
    # Step 1: Read raw data from the .dat file
    data = np.fromfile(data_file_path, dtype=np.int32)

    # Step 2: Convert the data into a complex number format
    data_I = data[0::2]  # Even-indexed items as I
    data_Q = data[1::2]  # Odd-indexed items as Q

    # Step 3: Handle two's complement for values greater than 32768
    data_I = np.where(data_I > 32768, data_I - 65536, data_I)
    data_Q = np.where(data_Q > 32768, data_Q - 65536, data_Q)

    # Step 4: Combine I and Q into complex numbers
    data_complex = data_I + 1j * data_Q

    # Step 5: Calculate the total number of samples expected
    total_samples = frame_length * num_frames
    print(f"Expected total samples: {total_samples}")
    print(f"Actual data length: {len(data_complex)}")

    # Step 6: Handle cases where the data is smaller or larger than expected
    if len(data_complex) < total_samples:
        print(f"Warning: Data length ({len(data_complex)}) is smaller than expected ({total_samples})")
        # Adjust num_frames to match the data size
        num_frames = len(data_complex) // frame_length
        print(f"Adjusted num_frames: {num_frames}")
        data_complex = data_complex[:num_frames * frame_length]  # Trim data to match the new num_frames
    elif len(data_complex) > total_samples:
        print(f"Warning: Data length ({len(data_complex)}) is larger than expected ({total_samples})")
        # Trim excess data if it's larger than expected
        data_complex = data_complex[:total_samples]
        print(f"Adjusted data length after trimming: {len(data_complex)}")

    # Step 7: Reshape the data
    # reshaped_data = data_complex.reshape(-1, frame_length)
    # print(f"Data shape after reshaping: {reshaped_data.shape}")

    # Step 8: Save the reshaped data to a .npy file for use with IQDataset
    np.save(save_path, data_complex)
    print(f"Data saved to {save_path} with shape {data_complex.shape}")


    return data_complex


def process_directory(root_dir):
    # 定义预处理输出根目录
    preprocess_root = "DATA/preprocessed"
    # Traverse through all subdirectories and process .dat files
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.dat'):
                data_file_path = os.path.join(subdir, file)
                print(f"Processing file: {data_file_path}")

                # 计算相对路径（保持原始目录结构）
                rel_path = os.path.relpath(subdir, root_dir)
                # 构建预处理目录路径
                preprocess_dir = os.path.join(preprocess_root, rel_path)
                os.makedirs(preprocess_dir, exist_ok=True)  # 创建目录（若不存在）

                # 构建保存路径
                device_name = os.path.splitext(file)[0]
                save_path = os.path.join(preprocess_dir, f"{device_name}.npy")

                # 调用转换函数并传入保存路径
                convert_to_IQ(data_file_path, save_path)

# 传入新数据集的根目录
root_dir = "DATA/new40classrawdata"
process_directory(root_dir)
