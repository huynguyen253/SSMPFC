import importlib
import os
import sys
import numpy as np

def normalize_multiview_data(views, method="zscore", axis=1, global_stats=False):
    normalized_views = []
    for view in views:
        if method == "zscore":
            # Normalize using mean and std for this view
            mean = np.nanmean(view, axis=axis, keepdims=True)
            std = np.nanstd(view, axis=axis, keepdims=True)
            normalized_view = (view - mean) / std
        elif method == "range":
            # Normalize using min and max for this view
            min_val = np.nanmin(view, axis=axis, keepdims=True)
            max_val = np.nanmax(view, axis=axis, keepdims=True)
            normalized_view = (view - min_val) / (max_val - min_val)
        normalized_views.append(normalized_view)
    return normalized_views

def min_max_normalize_columns(matrix):
    """
    Chuẩn hóa ma trận theo từng cột sao cho các giá trị của mỗi cột được đưa về 0 hoặc 1.

    Parameters:
        matrix (numpy.ndarray): Ma trận đầu vào.

    Returns:
        numpy.ndarray: Ma trận sau khi chuẩn hóa min-max theo cột.
    """    
    # Tìm giá trị min và max theo từng cột
    col_min = np.min(matrix, axis=0)
    col_max = np.max(matrix, axis=0)

    # Tránh chia cho 0 nếu cột có giá trị đồng nhất
    col_range = np.where(col_max - col_min == 0, 1, col_max - col_min)

    # Áp dụng công thức chuẩn hóa min-max
    normalized_matrix = (matrix - col_min) / col_range

    return normalized_matrix

def normalize_multiview_data2(view_c, views):
    normalized_views = []
    for view in views:
        view = np.array(view, dtype=float)
        normalized_views.append(min_max_normalize_columns(view))
    view_c = np.array(view_c, dtype=float)
    normalized_view_c = min_max_normalize_columns(view_c)
    return normalized_view_c, normalized_views

def normalize_multiview_data3(views):
    normalized_views = []
    for view in views:
        view = np.array(view, dtype=float)
        normalized_views.append(min_max_normalize_columns(view))
    return normalized_views
# PATH = os.path.dirname(os.path.abspath(__file__))
# MODULE_PATH = PATH + '/data/'
# sys.path.append(MODULE_PATH)
# module = importlib.import_module("UCI")
# multiview_data = module.X
# # Normalize each view globally using range normalization
# global_normalized_views = normalize_multiview_data(multiview_data, method="range", axis=1, global_stats=False)


# print("\nGlobal Normalization (Range):")
# print(global_normalized_views)
