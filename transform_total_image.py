#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import h5py
from scipy.io import loadmat, savemat

def read_mat(filepath):
    """Read MAT file and return its contents"""
    try:
        # First try normal scipy.io.loadmat
        data = loadmat(filepath)
        return data
    except Exception as e:
        # If that fails, try h5py approach for newer MAT files
        try:
            f = h5py.File(filepath, 'r')
            data = {}
            for k, v in f.items():
                data[k] = np.array(v)
            return data
        except Exception as e2:
            print(f"Error loading MAT file with h5py: {e2}")
            return None

def transform_total_image(input_filepath, output_filepath):
    """
    Biến đổi nhãn trong TotalImage:
    - Nhãn 0 (không ground truth) -> -1
    - Nhãn 1-7 -> giữ nguyên
    
    Args:
        input_filepath: Đường dẫn tới file TotalImage.mat gốc
        output_filepath: Đường dẫn tới file output (đã biến đổi)
    """
    print(f"Đọc file TotalImage từ: {input_filepath}")
    total_data = read_mat(input_filepath)
    
    if total_data is None:
        print("Không thể đọc file TotalImage. Kiểm tra lại đường dẫn.")
        return
    
    # Tìm key chính chứa dữ liệu
    total_key = None
    for key in total_data.keys():
        if not key.startswith('__'):  # Skip metadata
            total_key = key
            break
    
    if total_key is None:
        print("Không tìm thấy dữ liệu chính trong file TotalImage.")
        return
    
    # Truy cập dữ liệu
    labels = total_data[total_key]
    print(f"Dữ liệu gốc có kích thước: {labels.shape}")
    
    # Đếm số lượng điểm theo nhãn trước khi biến đổi
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\nPhân bố nhãn trước khi biến đổi:")
    for lbl, cnt in zip(unique_labels, counts):
        print(f"Nhãn {lbl}: {cnt} điểm")
    
    # Biến đổi nhãn: trừ tất cả các nhãn đi 1
    # Kết quả: 0 -> -1 (invalid), 1-7 -> 0-6 (valid labels)
    transformed_labels = labels.copy()
    
    # Chuyển đổi kiểu dữ liệu để tránh tràn số
    if transformed_labels.dtype.kind in ['u']:  # unsigned integer
        transformed_labels = transformed_labels.astype(np.int32)
    
    # Trừ tất cả các nhãn đi 1
    transformed_labels = transformed_labels - 1
    
    # Đếm số lượng điểm theo nhãn sau khi biến đổi
    unique_transformed, counts_transformed = np.unique(transformed_labels, return_counts=True)
    print("\nPhân bố nhãn sau khi biến đổi:")
    for lbl, cnt in zip(unique_transformed, counts_transformed):
        print(f"Nhãn {lbl}: {cnt} điểm")
    
    # Lưu kết quả
    print(f"\nLưu kết quả vào: {output_filepath}")
    output_data = {total_key: transformed_labels}
    savemat(output_filepath, output_data)
    
    print(f"Đã biến đổi thành công và lưu vào {output_filepath}")
    print(f"CHÚ Ý: Đã trừ tất cả các nhãn đi 1:")
    print(f"- Nhãn 0 -> -1 (invalid, không có ground truth)")
    print(f"- Nhãn 1-7 -> 0-6 (valid labels, có ground truth)")
    print(f"LƯU Ý: Đã chuyển đổi kiểu dữ liệu từ {labels.dtype} sang int32 để tránh tràn số")

if __name__ == "__main__":
    # Đường dẫn mặc định
    default_input = "D:/Download/Datasets/Datasets/HS-SAR-DSM Augsburg/TotalImage.mat"
    default_output = "D:/Download/Datasets/Datasets/HS-SAR-DSM Augsburg/TotalImage_transformed.mat"
    
    # Kiểm tra nếu có tham số dòng lệnh
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_path = sys.argv[2] if len(sys.argv) > 2 else default_output
    
    transform_total_image(input_path, output_path)