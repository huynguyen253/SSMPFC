import numpy as np
from scipy.linalg import svd
import time

def initialize_matrices(X_views, num_anchors, num_samples):
    """
    Khởi tạo các ma trận với kích thước chính xác.
    """
    num_views = len(X_views)    
    B = [np.random.rand(X.shape[1], num_anchors) for X in X_views]  # (num_features, num_anchors)
    H = [np.random.rand(num_anchors, num_samples) for _ in X_views]  # (num_anchors, num_samples)
    A_common = np.random.rand(num_anchors, num_anchors)  # (num_anchors, num_anchors)
    A_specific = [np.random.rand(num_anchors, num_anchors) for _ in range(num_views)]  # (num_anchors, num_anchors)
    Z_common = np.random.rand(num_anchors, num_samples)  # (num_anchors, num_samples)
    Z_specific = [np.random.rand(num_anchors, num_samples) for _ in X_views]  # (num_anchors, num_samples)
    return B, H, A_common, A_specific, Z_common, Z_specific

def update_B(X, H):
    """
    Cập nhật ma trận B.
    """
    return X.T @ H.T @ np.linalg.inv(H @ H.T)

def update_H(X, B, A_common, Z_common, A_specific, Z_specific):
    """
    Cập nhật ma trận H.
    """
    # Kiểm tra kích thước
    if B.T.shape[1] != X.T.shape[0]:
        raise ValueError(f"Mismatch in B.T and X.T dimensions: {B.T.shape}, {X.T.shape}")
    if A_common.shape[0] != Z_common.shape[0]:
        raise ValueError(f"Mismatch in A_common/Z_common dimensions: {A_common.shape}, {Z_common.shape}")
    if A_specific.shape[0] != Z_specific.shape[0]:
        raise ValueError(f"Mismatch in A_specific/Z_specific dimensions: {A_specific.shape}, {Z_specific.shape}")

    # Tính các thành phần
    term1 = B.T @ X.T  # (num_anchors, num_samples)
    term2 = A_common @ Z_common  # (num_anchors, num_samples)
    term3 = A_specific @ Z_specific  # (num_anchors, num_samples)

    # Kiểm tra tính tương thích trước phép cộng
    if term1.shape != term2.shape or term1.shape != term3.shape:
        raise ValueError(f"Mismatch in term shapes: term1 {term1.shape}, term2 {term2.shape}, term3 {term3.shape}")

    # Tổng hợp
    H_new = np.linalg.inv(B.T @ B + np.eye(B.shape[1])) @ (term1 + term2 + term3)
    return H_new

def update_A(Z, H):
    """
    Cập nhật ma trận A (chung hoặc cụ thể).
    """
    E = H @ Z.T
    U, _, Vt = svd(E)
    return U @ Vt

def update_Z(H, A, alpha, Z_constraints=True):
    """
    Cập nhật ma trận Z (chung hoặc cụ thể).
    - H: Ma trận ẩn của một view (num_anchors, num_samples).
    - A: Ma trận neo (num_anchors, num_anchors).
    - alpha: Tham số điều chỉnh.
    - Z_constraints: Có chuẩn hóa Z hay không.
    """
    # Kiểm tra kích thước
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square: {A.shape}")
    if A.shape[1] != H.shape[0]:
        raise ValueError(f"Mismatch in A and H dimensions: A {A.shape}, H {H.shape}")

    # Tính toán Z
    G = A.T @ H / (1 + alpha)  # (num_anchors, num_samples)
    
    if Z_constraints:
        # Chuẩn hóa Z theo cột
        Z_new = np.maximum(G, 0)
        Z_new /= np.sum(Z_new, axis=0, keepdims=True)  # Normalize columns
    else:
        Z_new = G
    
    return Z_new



def MV_DAGL(X_views, num_anchors, max_iter=100, alpha=0.2, tol=1e-6):
    """
    Thuật toán MV_DAGL.
    """
    num_views = len(X_views)
    num_samples = X_views[0].shape[0]
    B, H, A_common, A_specific, Z_common, Z_specific = initialize_matrices(X_views, num_anchors,num_samples)
    obj_old = float('inf')

    for t in range(max_iter):
        # Cập nhật Z_specific
        Z_specific = [update_Z(H[k], A_specific[k], alpha) for k in range(num_views)]

        # Cập nhật A_specific
        A_specific = [update_A(Z_specific[k], H[k]) for k in range(num_views)]

        # Cập nhật Z_common
        H_mean = np.mean(H, axis=0)  # Trung bình của H dọc theo các view
        Z_common = update_Z(H_mean, A_common, alpha)
        # Z_common_all = update_Z(H_mean, A_common, alpha)

        # Cập nhật A_common
        A_common = update_A(Z_common, H_mean)

        # print("matrix:")
        # for l in range(num_views):
        #     print("view ",l)
        #     print(Z_specific[l].shape)
        #     print(A_specific[l].shape)
        # print(A_common.shape)
        # print(Z_common.shape)
        # time.sleep(5)

        # Cập nhật H
        H = [update_H(X_views[k], B[k], A_common, Z_common, A_specific[k], Z_specific[k]) for k in range(num_views)]

        # print("matrix:")
        # for l in range(num_views):
        #     print("view ",l)
        #     print(H[l].shape)
        #     print(X_views[0].shape)
        # time.sleep(5)


        # Cập nhật B
        B = [update_B(X_views[k], H[k]) for k in range(num_views)]

        # Tính giá trị hàm mục tiêu
        obj_new = sum(np.linalg.norm(X_views[k] - (B[k] @ H[k]).T, 'fro')**2 for k in range(num_views)) \
                  + alpha * (np.linalg.norm(Z_common, 'fro')**2 + sum(np.linalg.norm(Z_specific[k], 'fro')**2 for k in range(num_views)))

        # Kiểm tra hội tụ
        if abs(obj_old - obj_new) < tol:
            break
        obj_old = obj_new

    return Z_common, Z_specific

