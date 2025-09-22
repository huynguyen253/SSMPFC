import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

def compute_cosine_similarity_with_pca(X, Y, n_components=None):
    """
    Tính ma trận cosine similarity giữa hai ma trận X và Y sau khi giảm chiều bằng PCA.

    Parameters:
    - X: numpy array, kích thước (m, n_x) - ma trận dữ liệu đầu tiên.
    - Y: numpy array, kích thước (m, n_y) - ma trận dữ liệu thứ hai.
    - n_components: Số chiều giảm chung, nếu None thì chọn số chiều nhỏ hơn giữa X và Y.

    Returns:
    - S: Ma trận cosine similarity (numpy array).
    """
    # Nếu cần, chọn số chiều chung
    if n_components is None:
        n_components = min(X.shape[1], Y.shape[1])

    # Giảm chiều của cả hai ma trận về cùng số chiều
    pca_X = PCA(n_components=n_components)
    X_reduced = pca_X.fit_transform(X)

    pca_Y = PCA(n_components=n_components)
    Y_reduced = pca_Y.fit_transform(Y)

    # Tính cosine similarity giữa hai ma trận đã giảm chiều
    S = cosine_similarity(X_reduced, Y_reduced)
    return S

def create_augmented_matrix(views, n_components=None):
    """
    Tính ma trận augmented X_alpha dựa trên cosine similarity giữa các view.
    
    Parameters:
    - views: List các numpy arrays, mỗi array là một view (kích thước: m x n_v).
    - n_components: Số lượng thành phần PCA giữ lại, nếu None thì giữ nguyên tất cả chiều.
    
    Returns:
    - X_alpha: Ma trận augmented (numpy array).
    """
    # Số lượng view và số mẫu
    num_views = len(views)
    num_samples = views[0].shape[0]
    
    # Giảm chiều từng view bằng PCA
    reduced_views = []
    for view in views:
        if n_components is not None and n_components < view.shape[1]:
            pca = PCA(n_components=n_components)
            reduced_view = pca.fit_transform(view)
        else:
            reduced_view = view  # Không giảm chiều nếu không cần
        reduced_views.append(reduced_view)
    
    # Xác định tổng số chiều của ma trận augmented
    total_features = sum(view.shape[1] for view in reduced_views)
    augmented_matrix = np.zeros((num_samples, total_features))  # Kích thước: m x tổng số chiều
    
    # Xây dựng ma trận augmented
    current_col = 0  # Vị trí cột hiện tại trong ma trận augmented
    for p in range(num_views):
        # Gán phần tử trên đường chéo: chính ma trận view
        num_features_p = reduced_views[p].shape[1]
        augmented_matrix[:, current_col:current_col + num_features_p] = reduced_views[p]
        
        # Xử lý phần tử ngoài đường chéo
        for q in range(num_views):
            if p != q:
                # Tính cosine similarity với PCA
                S_pq = compute_cosine_similarity_with_pca(reduced_views[p], reduced_views[q], n_components=n_components)
                X_pq = S_pq @ reduced_views[p] 
                
                # Thêm ma trận tương quan vào các cột tiếp theo (không overwrite)
                augmented_matrix[:, current_col:current_col + num_features_p] += X_pq
        
        current_col += num_features_p  # Cập nhật vị trí cột tiếp theo
    
    return augmented_matrix


def create_augmented_matrix_bk(views, add_bias=False, compute_cosine=True):
    """
    Tạo ma trận augmented X_alpha từ danh sách các view và tính cosine similarity.
    
    Parameters:
    - views: List of numpy arrays hoặc pandas DataFrames, mỗi phần tử là một view với cùng số lượng bản ghi.
    - add_bias: Boolean, nếu True sẽ thêm cột độ chệch (bias term).
    - compute_cosine: Boolean, nếu True sẽ tính cosine similarity và thêm vào ma trận.
    
    Returns:
    - X_alpha: Pandas DataFrame là ma trận augmented.
    """
    df_views = []
    avg_cos_sim = []
    
    for i, view in enumerate(views):
        if isinstance(view, np.ndarray):
            # Tạo tên cột tự động nếu view là numpy array
            n_features = view.shape[1]
            columns = [f'V{i+1}_F{j+1}' for j in range(n_features)]
            df = pd.DataFrame(view, columns=columns)
        elif isinstance(view, pd.DataFrame):
            df = view
        else:
            raise ValueError("Các view phải là numpy arrays hoặc pandas DataFrames.")
        
        if compute_cosine:
            # Tính cosine similarity matrix
            cos_sim = cosine_similarity(view)
            # Tính trung bình cosine similarity cho mỗi mẫu
            avg_cos = cos_sim.mean(axis=1)
            # Thêm vào DataFrame với tên cột phù hợp
            df[f'V{i+1}_avg_cos_sim'] = avg_cos
        
        df_views.append(df)
    
    # Nối các view lại với nhau
    X_alpha = pd.concat(df_views, axis=1)
    
    # Thêm cột Bias nếu cần
    if add_bias:
        X_alpha.insert(0, 'Bias', 1)
    
    return X_alpha

# # Sử dụng hàm
# np.random.seed(0)
# # Giả sử bạn có 3 góc nhìn với các số lượng thuộc tính khác nhau
# view1 = np.random.rand(100, 50)  # Góc nhìn 1: 100 mẫu, 50 thuộc tính
# view2 = np.random.rand(100, 30)  # Góc nhìn 2: 100 mẫu, 30 thuộc tính
# view3 = np.random.rand(100, 20)  # Góc nhìn 3: 100 mẫu, 20 thuộc tính

# multiview_data = [view1, view2, view3]

# X_alpha = create_augmented_matrix(views, add_bias=True, compute_cosine=True)

# print("Ma trận augmented X_alpha:")
# print(X_alpha)
