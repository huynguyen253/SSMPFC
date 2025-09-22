import numpy as np
import sys
import os
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score, davies_bouldin_score
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from fcmeans import FCM
from scipy.linalg import svd
from config import purity_score, clustering_accuracy
from sklearn.decomposition import PCA
from normalization import normalize_multiview_data3

def preprocessData(filename,data):
    modules = [
        ("ACM", 3), 
        ("ALOI", 100), 
        # ("BBCSport", 5), 
        # ("Caltech", 102),
        # ("CiteSeer", 6), 
        # ("Cora", 7), 
        # ("Handwritten", 10),
        # ("mul_ORL", 40), 
        # ("MNIST4", 4), 
        # ("MNIST10k", 10),
        # ("Movies", 17), 
        ("MSRC-v5", 7), 
        # ("NUS-WIDE31", 12),
        # ("OutdoorScene", 8), 
        # ("Prokaryotic", 4), 
        # ("ProteinFold", 27),
        # ("Reuters-1200", 6), 
        ("3Sources", 6),
        # ("UCI", 10), 
        # ("WebKB", 2), 
        # ("Wikipedia", 10),
        ("Yale", 15, 169, 3, [9,50,512]), 
        # ("Wikipedia-test", 10), 
        ("Caltech101-20", 20)
        #, ("Reuters-1500", 12)
        ]

def direct_accuracy(y_true, y_pred):
    """Calculate accuracy directly without cluster permutation"""
    return np.mean(y_true == y_pred)

def validity(l, label, predict, filename, is_training=False):
    """Calculate validity measures and save to file
    
    Args:
        l: view index
        label: true labels
        predict: predicted labels
        filename: output file path
        is_training: whether evaluating on training set (labeled data)
    """
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    f = open(filename, "w")
    
    # For training data (labeled), use direct accuracy
    if is_training:
        acc = direct_accuracy(label, predict)
    else:
        # For testing data (unlabeled), use clustering accuracy with permutation
        acc = clustering_accuracy(label, predict)
        
    # Calculate other metrics
    ari = adjusted_rand_score(label, predict)
    nmi = normalized_mutual_info_score(label, predict)
    f1 = f1_score(label, predict, average='weighted')
    pur = purity_score(label, predict)
    
    # Calculate Kappa coefficient
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(label, predict)
    
    # Print results to console
    print(f"\nClustering Metrics:")
    print(f"ACC: {acc:.4f}")
    print(f"ARI: {ari:.4f}")
    print(f"NMI: {nmi:.4f}")
    print(f"F1:  {f1:.4f}")
    print(f"PUR: {pur:.4f}")
    print(f"Kappa: {kappa:.4f}")
    
    # Write results to file
    f.write(f"ACC: {acc:.4f}\n")
    f.write(f"ARI: {ari:.4f}\n")
    f.write(f"NMI: {nmi:.4f}\n")
    f.write(f"F1:  {f1:.4f}\n")
    f.write(f"PUR: {pur:.4f}\n")
    f.write(f"Kappa: {kappa:.4f}\n")
    
    # Write detailed results
    f.write("\nDetailed Results:\n")
    f.write("Real labels:\n")
    f.write(",".join(map(str, label)) + "\n")
    f.write("Predicted labels:\n")
    f.write(",".join(map(str, predict)) + "\n")
    
    f.close()
    
    return acc, ari, nmi, f1, pur, kappa

def calculate_fcm_all(data,c):
    dataall = np.hstack(data)
    fcm = FCM(n_clusters=c)
    fcm.fit(dataall)
    return fcm.u.T, fcm.centers

def calculate_dissimilarity(u_fcm, L, c, N):
    dis = np.zeros(((L + 1) * c, (L + 1) * c, N))
    for k in range(N):
        for l in range(L + 1):
            for j in range(c):    
                for h in range(L + 1):
                    for i in range(c):
                        dis[l*c + j][h*c + i][k] = (u_fcm[l][j][k] - u_fcm[h][i][k]) ** 2
    return dis

def calculate_fcm(data,c):
    u = []
    v = []
    for dat in data:
        fcm = FCM(n_clusters=c)
        fcm.fit(dat)
        u.append(fcm.u.T)
        v.append(np.array(fcm.centers))
    return u, v

def calculate_anchor_graph(data, L, n_anchors):
    print("Begin calculate anchor graph")
    dat = []
    rel = []
    for l in range(L): 
        fcm = FCM(n_clusters=n_anchors)
        fcm.fit(np.array(data[l]))
        dat.append(fcm.centers)
        rel.append(fcm.u)

    print("Begin calculate common anchor graph")
    dataall = np.hstack(data)
    fcm = FCM(n_clusters=n_anchors)
    fcm.fit(dataall)
    rel.append(fcm.u)
    dat.append(fcm.centers)
    return normalize_multiview_data3(dat), rel

def calculate_similarity(data, L, n_anchors, c):    
    rep = []
    simi = []
    for l in range(L): 
        fcm = FCM(n_clusters=n_anchors)
        fcm.fit(np.array(data[l]))
        re = np.copy(fcm.u)
        rep.append(re)

    dataall = np.hstack(data)
    fcm = FCM(n_clusters=n_anchors)
    fcm.fit(dataall)
    re = np.copy(fcm.u)
    rep.append(re)

    # Chuyển ma trận sang dạng numpy array nếu chưa phải
    rep = np.array(rep, dtype=float)    
    for l in range(L):
        si = rep[l] @ rep[l].T
        simi.append(si)
    rep = normalize_multiview_data3(rep)
    # simi = normalize_multiview_data3(simi)
    return rep, simi

def calculate_similarity_bk(data, L, n_anchors, c):    
    print("calculate anchors graph")
    dat, rel = calculate_anchor_graph(data, L, n_anchors)
    print("clustering anchors graph")
    rep = []
    simi = []
    for l in range(L): 
        fcm = FCM(n_clusters=c)
        fcm.fit(np.array(dat[l]))
        re = np.copy(fcm.u)
        rep.append(re)

    # Chuyển ma trận sang dạng numpy array nếu chưa phải
    rep = np.array(rep, dtype=float)
    repre = normalize_multiview_data3(rep)
    for l in range(L):
        si = repre[l] @ repre[l].T
        simi.append(si)
    # simi = normalize_multiview_data3(simi) 

    return repre, simi, rel

def dual_anchor_graph_similarity(data, n_anchors=10, sigma=1.0):
    """
    Tính ma trận similarity dựa trên Dual Anchor Graph Learning (DAGL).
    
    Args:
        X (np.ndarray): Dữ liệu đầu vào, kích thước (n_samples, n_features).
        n_anchors (int): Số lượng anchor points.
        sigma (float): Tham số điều chỉnh độ mượt của Gaussian kernel.
    
    Returns:
        np.ndarray: Ma trận similarity (n_samples, n_samples).
    """
    similarity_matrix = []
    for X in data:
        n_samples = len(X)
        n_features = len(X[0])

        # Bước 1: Chọn các anchor points bằng KMeans
        # kmeans = KMeans(n_clusters=n_anchors, random_state=42)
        # kmeans.fit(X)
        # anchors = kmeans.cluster_centers_

        fcm = FCM(n_clusters=n_anchors)
        fcm.fit(X)
        # outputs
        anchors = fcm.centers

        # Bước 2: Tính khoảng cách Gaussian giữa mỗi điểm dữ liệu và các anchor
        distances = euclidean_distances(X, anchors)  # Kích thước (n_samples, n_anchors)
        W_a = np.exp(-distances**2 / (2 * sigma**2))

        # Chuẩn hóa hàng của ma trận W_a
        W_a = W_a / W_a.sum(axis=1, keepdims=True)

        # Bước 3: Tính ma trận similarity
        # S = W_a * W_a^T
        similarity_matrix.append(W_a @ W_a.T)
    return similarity_matrix


def compute_multiview_similarity(X):
    """
    Tính ma trận similarity của dữ liệu multiview.
    
    Args:
        X (list of np.ndarray): Danh sách các ma trận dữ liệu, mỗi phần tử tương ứng với một view.
                               Kích thước mỗi ma trận là (n_samples, n_features).
    
    Returns:
        np.ndarray: Ma trận similarity cuối cùng, kích thước (n_samples, n_samples).
    """
    # Kiểm tra dữ liệu đầu vào
    if not isinstance(X, list) or len(X) == 0:
        raise ValueError("X phải là một danh sách chứa các ma trận numpy.")
    
    # Lấy số lượng mẫu từ view đầu tiên
    n_samples = len(X[0])
    
    # Ma trận similarity tổng hợp
    combined_similarity = np.zeros((n_samples, n_samples))
    
    # Tính similarity cho từng view và cộng vào combined_similarity
    for view in X:
        if len(view) != n_samples:
            raise ValueError("Tất cả các view phải có cùng số lượng mẫu.")
        similarity = cosine_similarity(view)
        combined_similarity += similarity
    
    # Trung bình hóa similarity
    combined_similarity /= len(X)    
    return combined_similarity

def preprocess_views(X_list, n_components):
    """
    Biến đổi tất cả các view về cùng số chiều bằng PCA.
    
    Args:
        X_list (list of np.ndarray): Danh sách các view.
        n_components (int): Số chiều đầu ra cố định.
        
    Returns:
        list of np.ndarray: Danh sách các view đã được giảm chiều.
    """
    pca = PCA(n_components=n_components)
    return [pca.fit_transform(X) for X in X_list]

def dual_anchor_graph_multiview(X_list, n_anchors=10, sigma=1.0):
    """
    Tính ma trận Common Anchor Graph và Specific Anchor Graph từ dữ liệu multiview.
    
    Args:
        X_list (list of np.ndarray): Danh sách các ma trận dữ liệu từ các view.
                                     Mỗi ma trận có kích thước (n_samples, n_features).
        n_anchors (int): Số lượng anchor points.
        sigma (float): Tham số điều chỉnh độ mượt của Gaussian kernel.
    
    Returns:
        tuple: 
            - S_common (np.ndarray): Ma trận Common Anchor Graph (n_samples, n_samples).
            - S_specific_list (list of np.ndarray): Danh sách các ma trận Specific Anchor Graph 
                                                    cho từng view (n_samples, n_samples).
    """
    # Kiểm tra dữ liệu đầu vào
    if not isinstance(X_list, list) or len(X_list) == 0:
        raise ValueError("X_list phải là một danh sách chứa các ma trận numpy.")
    
    n_samples = len(X_list[0])
    n_views = len(X_list)

    nsize = np.zeros((n_views),dtype=int)

    for i in range(n_views):
        nsize[i] = int(len(X_list[i][0]))

    # Bước 1: Chọn các anchor points chung từ tất cả các view
    combined_X = np.hstack(X_list)  # Ghép các view lại theo cột
    
    fcm = FCM(n_clusters=n_anchors)
    fcm.fit(combined_X)
    # outputs
    anchors = fcm.centers    

    # Bước 2: Tính ma trận similarity cho từng view và cho toàn bộ
    S_common = np.zeros((n_samples, n_samples))
    S_specific_list = []
    
    for i in range(n_views):
        view = X_list[i]
        
        # prepare anchor for view
        idx_start = 0
        for j in range(0,i):
            idx_start += nsize[j]

        anchor_s = np.zeros((n_anchors,nsize[i]),dtype=float)
        for ia in range(n_anchors):
            for j in range(nsize[i]):
                anchor_s[ia][j] = anchors[ia][idx_start + j]       

        # Tính khoảng cách từ các điểm trong view đến các anchor
        distances = euclidean_distances(view, anchor_s)  # (n_samples, n_anchors)
        W_a = np.exp(-distances**2 / (2 * sigma**2))

        # Chuẩn hóa W_a theo hàng
        W_a = W_a / W_a.sum(axis=1, keepdims=True)

        # Tính ma trận similarity cho view hiện tại
        S_view = W_a @ W_a.T

        # Cộng dồn vào ma trận common similarity
        S_common += S_view
        
        # Lưu lại ma trận specific similarity cho view hiện tại
        S_specific_list.append(S_view)
    
    # Trung bình hóa ma trận Common Anchor Graph
    S_common /= n_views
    
    return [S_common, S_specific_list]



def compute_latent_representation(S, n_components=10):
    """
    Tính latent representation từ ma trận similarity sử dụng SVD.
    
    Args:
        S (np.ndarray): Ma trận similarity (n_samples, n_samples).
        n_components (int): Số chiều không gian ẩn cần trích xuất.
    
    Returns:
        np.ndarray: Latent representation (n_samples, n_components).
    """
    # Bước 1: Phân rã SVD
    U, Sigma, Vt = svd(S)
    
    # Bước 2: Trích xuất các latent representation
    latent_representation = U[:, :n_components] @ np.diag(Sigma[:n_components])
    
    return latent_representation

def calc_representation(S_specific_list,n_components):
        
    # Tính latent representation từ từng ma trận Specific Similarity
    latent_specific_list = [
        compute_latent_representation(S_specific, n_components=n_components)
        for S_specific in S_specific_list
    ]
    return latent_specific_list

def multiview_davies_bouldin_index(X_views, labels):
    """
    Calculate multiview Davies-Bouldin index.
    
    Args:
        X_views: List of data matrices for each view
        labels: Cluster assignments
    Returns:
        Multiview Davies-Bouldin index
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    n_views = len(X_views)
    db_scores = []
    
    for X in X_views:
        # Calculate cluster centers
        centers = np.array([X[labels == i].mean(axis=0) for i in unique_labels])
        
        # Calculate average distances within clusters
        cluster_distances = []
        for i in range(n_clusters):
            cluster_points = X[labels == unique_labels[i]]
            if len(cluster_points) > 0:
                distances = np.mean(np.sqrt(((cluster_points - centers[i]) ** 2).sum(axis=1)))
                cluster_distances.append(distances)
            else:
                cluster_distances.append(0)
        cluster_distances = np.array(cluster_distances)
        
        # Calculate Davies-Bouldin index
        db_score = 0
        for i in range(n_clusters):
            if cluster_distances[i] == 0:
                continue
            max_ratio = 0
            for j in range(n_clusters):
                if i != j and cluster_distances[j] != 0:
                    center_distance = np.sqrt(((centers[i] - centers[j]) ** 2).sum())
                    ratio = (cluster_distances[i] + cluster_distances[j]) / center_distance
                    max_ratio = max(max_ratio, ratio)
            db_score += max_ratio
        
        db_score /= n_clusters
        db_scores.append(db_score)
    
    return np.mean(db_scores)