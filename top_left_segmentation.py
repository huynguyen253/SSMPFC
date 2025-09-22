# Top-Left Part Segmentation with MPFCM
import numpy as np
import os
import scipy.io as sio
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time
import random
import json
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import math
import h5py
import warnings
from sklearn.preprocessing import normalize
from scipy.linalg import svd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from mpfcm import main_MPFCM, run_MPFCM
from MV_DAGL import MV_DAGL
from normalization import normalize_multiview_data3

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
            # Create dummy data
            return {"dummy_data": np.zeros((332, 485))}

def load_augsburg_data(dsm_filepath="D:/Download/Datasets/Datasets/HS-SAR-DSM Augsburg/data_DSM.mat", 
                        hs_filepath="D:/Download/Datasets/Datasets/HS-SAR-DSM Augsburg/data_HS_LR.mat", 
                        sar_filepath="D:/Download/Datasets/Datasets/HS-SAR-DSM Augsburg/data_SAR_HR.mat",
                        total_filepath="D:/Download/Datasets/Datasets/HS-SAR-DSM Augsburg/TotalImage_transformed.mat"):
    """
    Load the Augsburg dataset using TotalImage for labels.
    
    Returns:
        hs_data: Hyperspectral data
        sar_data: SAR data
        dsm_data: DSM data
        labels: Ground truth labels from TotalImage
        valid_mask: Mask for all valid points (where label > 0)
    """
    print(f"\nLoading Augsburg data...")
    
    # Load DSM data (Digital Surface Model)
    print(f"Loading DSM data from {dsm_filepath}...")
    try:
        data_DSM = read_mat(dsm_filepath)
        dsm = data_DSM["data_DSM"]  # Shape (332, 485)
        print(f"DSM data shape: {dsm.shape}")
    except Exception as e:
        print(f"Error loading DSM data: {e}")
        print("Using dummy DSM data")
        dsm = np.random.rand(332, 485)
    
    # Load Hyperspectral data
    print(f"Loading HS data from {hs_filepath}...")
    try:
        data_HS_LR = read_mat(hs_filepath)
        hs = data_HS_LR["data_HS_LR"]  # Shape (332, 485, 180)
        print(f"HS data shape: {hs.shape}")
    except Exception as e:
        print(f"Error loading HS data: {e}")
        print("Using dummy HS data")
        hs = np.random.rand(332, 485, 180)
    
    # Load SAR data
    print(f"Loading SAR data from {sar_filepath}...")
    try:
        data_SAR_HR = read_mat(sar_filepath)
        sar = data_SAR_HR["data_SAR_HR"]  # Shape (332, 485, 4)
        print(f"SAR data shape: {sar.shape}")
    except Exception as e:
        print(f"Error loading SAR data: {e}")
        print("Using dummy SAR data")
        sar = np.random.rand(332, 485, 4)
    
    # Load TotalImage for ground truth labels
    print(f"Loading label data from {total_filepath}...")
    try:
        total_data = read_mat(total_filepath)
        
        # Find the main data key
        total_key = None
        for key in total_data.keys():
            if not key.startswith('__'):  # Skip metadata
                total_key = key
                break
                
        if total_key:
            labels = total_data[total_key]
            print(f"Labels shape: {labels.shape}")
            print(f"Using key '{total_key}' from TotalImage_transformed.mat")
        else:
            raise KeyError("No valid data key found in TotalImage_transformed.mat")
    except Exception as e:
        print(f"Error loading TotalImage data: {e}")
        print("Using dummy label data")
        labels = np.zeros((332, 485), dtype=int)
        # Randomly select some points for labels
        total_pixels = 332 * 485
        labeled_indices = np.random.choice(total_pixels, size=78000, replace=False)
        labeled_rows, labeled_cols = np.unravel_index(labeled_indices, (332, 485))
        labels[labeled_rows, labeled_cols] = np.random.randint(1, 8, size=78000)
    
    # Create mask for valid points (chỉ nhãn từ 0 đến 6 là valid, nhãn -1 là invalid)
    valid_mask = (labels >= 0)
    
    # Count valid points
    valid_points = np.sum(valid_mask)
    total_pixels = labels.size
    print(f"Total pixels: {total_pixels}")
    print(f"Valid labeled points (label > 0): {valid_points} ({valid_points/total_pixels*100:.1f}%)")
    
    # Print statistics about classes
    unique_classes = np.unique(labels[valid_mask])
    print(f"\nUnique classes in labels: {unique_classes}")
    print(f"Number of classes: {len(unique_classes)}")
    
    # Print distribution of classes
    print("\nClass distribution:")
    for cls in unique_classes:
        count = np.sum(labels == cls)
        print(f"  Class {cls}: {count} pixels ({count/valid_points*100:.2f}%)")
    
    return hs, sar, dsm, labels, valid_mask

def extract_top_left_part(hs_data, sar_data, dsm_data, labels, n_rows=2, n_cols=2):
    """
    Extract only the top-left part from the full image.
    
    Args:
        hs_data: Full hyperspectral data
        sar_data: Full SAR data  
        dsm_data: Full DSM data
        labels: Full labels
        n_rows: Number of rows for division
        n_cols: Number of columns for division
        
    Returns:
        hs_part: Top-left hyperspectral part
        sar_part: Top-left SAR part
        dsm_part: Top-left DSM part
        labels_part: Top-left labels part
        indices: (start_h, end_h, start_w, end_w) for this part
    """
    height, width = hs_data.shape[:2]
    
    # Calculate part dimensions
    part_height = height // n_rows
    part_width = width // n_cols
    
    print(f"Full image dimensions: {height}x{width}")
    print(f"Part dimensions: {part_height}x{part_width}")
    
    # Top-left part indices (i=0, j=0)
    start_h = 0
    end_h = part_height
    start_w = 0
    end_w = part_width
    
    print(f"Top-left part indices: start_h={start_h}, end_h={end_h}, start_w={start_w}, end_w={end_w}")
    
    # Extract top-left parts
    hs_part = hs_data[start_h:end_h, start_w:end_w]
    sar_part = sar_data[start_h:end_h, start_w:end_w]
    dsm_part = dsm_data[start_h:end_h, start_w:end_w]
    labels_part = labels[start_h:end_h, start_w:end_w]
    
    print(f"Top-left part shapes:")
    print(f"  HS: {hs_part.shape}")
    print(f"  SAR: {sar_part.shape}")
    print(f"  DSM: {dsm_part.shape}")
    print(f"  Labels: {labels_part.shape}")
    
    indices = (start_h, end_h, start_w, end_w)
    
    return hs_part, sar_part, dsm_part, labels_part, indices

def prepare_data_for_processing(hs_part, sar_part, dsm_part, labels_part):
    """Prepare the data for processing algorithms with Augsburg-specific handling"""
    height, width = hs_part.shape[:2]
    
    print(f"\nPreparing data for processing...")
    print(f"Part dimensions: {height}x{width}")
    
    # Reshape to 2D for processing (pixels x features)
    hs_reshaped = hs_part.reshape(-1, hs_part.shape[-1])  # (height*width, bands)
    sar_reshaped = sar_part.reshape(-1, sar_part.shape[-1])  # (height*width, bands)
    dsm_reshaped = dsm_part.reshape(-1, 1)  # (height*width, 1)
    
    # Create position matrix for each pixel
    pixel_positions = np.zeros((height, width, 2), dtype=int)
    for i in range(height):
        for j in range(width):
            pixel_positions[i, j] = [i, j]
    pixel_positions_flat = pixel_positions.reshape(-1, 2)
    
    # Filter valid points if we have labels
    if labels_part is not None:
        labels_flat = labels_part.flatten()
        valid_indices = labels_flat != -1  # Lọc điểm KHÁC -1
        
        # Count and report
        total_points = len(labels_flat)
        valid_points = np.sum(valid_indices)
        filtered_points = total_points - valid_points
        
        print(f"Filtering unlabeled points...")
        print(f"Total points: {total_points}")
        print(f"Valid labeled points: {valid_points} ({valid_points/total_points*100:.1f}%)")
        print(f"Filtered unlabeled points: {filtered_points} ({filtered_points/total_points*100:.1f}%)")
        
        # Filter data
        hs_reshaped = hs_reshaped[valid_indices]
        sar_reshaped = sar_reshaped[valid_indices]
        dsm_reshaped = dsm_reshaped[valid_indices]
        labels_valid = labels_flat[valid_indices]
        pixel_positions_valid = pixel_positions_flat[valid_indices]
        
        # Create mask for mapping back
        mask = np.zeros(labels_flat.shape, dtype=bool)
        mask[valid_indices] = True
        mask_2d = mask.reshape(labels_part.shape)
    else:
        mask_2d = np.ones((height, width), dtype=bool)
        labels_valid = None
        pixel_positions_valid = pixel_positions_flat
    
    # Normalize data
    scaler_hs = StandardScaler()
    scaler_sar = StandardScaler()
    scaler_dsm = StandardScaler()
    
    hs_normalized = scaler_hs.fit_transform(hs_reshaped)
    sar_normalized = scaler_sar.fit_transform(sar_reshaped)
    dsm_normalized = scaler_dsm.fit_transform(dsm_reshaped)
    
    # Format data for algorithms (3 views: HS, SAR, DSM)
    data_views = [hs_normalized, sar_normalized, dsm_normalized]
    
    print(f"Data views shapes:")
    for i, view in enumerate(data_views):
        print(f"  View {i}: {view.shape}")
    
    return data_views, (height, width), mask_2d, labels_valid, pixel_positions_valid

def process_top_left_part(data_views, labels, c, n_anchors, params):
    """Process the top-left image part"""
    L = len(data_views)  # Number of views
    N = len(data_views[0])  # Number of samples
    
    print(f"\nProcessing top-left part with {N} samples and {L} views")
    
    # Normalize multiview data
    data = normalize_multiview_data3(data_views)
    
    print("Data after normalization:")
    for i, view in enumerate(data):
        print(f"View {i} shape: {view.shape}")
    
    # Sử dụng MV_DAGL để tạo biểu diễn chung và riêng
    print("Running MV-DAGL...")
    start_time = time.time()
    Z_common, Z_specific = MV_DAGL(data, n_anchors, max_iter=100, alpha=0.2, tol=1e-6)
    end_time = time.time()
    print(f"MV-DAGL completed in {end_time - start_time:.2f} seconds")
    
    print(f"Z_common shape: {Z_common.shape}")
    for i, z in enumerate(Z_specific):
        print(f"Z_specific[{i}] shape: {z.shape}")
    
    # Calculate similarity matrices and prepare data for MPFCM
    print("Calculating similarity matrices...")
    data_new = []
    similarity = []
    
    for l in range(L):
        tg = np.array(Z_specific[l].T @ Z_specific[l])
        similarity.append(tg)
        U, _, _ = svd(Z_specific[l].T, full_matrices=False)
        data_new.append(U[:, :n_anchors])

    # Add common representation
    U, _, _ = svd(Z_common.T, full_matrices=False)
    data_new.append(U[:, :n_anchors])

    tg = np.array(Z_common.T @ Z_common)
    similarity.append(tg)
    
    # Kiểm tra dữ liệu đầu vào
    print(f"Number of views (L+1): {len(data_new)}")
    print(f"Data shapes: {[d.shape for d in data_new]}")
    print(f"Number of similarity matrices: {len(similarity)}")
    
    # QUAN TRỌNG: Xử lý nhãn đúng cách cho semi-supervised learning
    if labels is not None:
        print(f"\nNhãn sau khi lọc từ prepare_data_for_processing:")
        print(f"- Tổng số điểm: {N}")
        print(f"- Tất cả {N} điểm đều có nhãn valid (>= 0)")
        print(f"- Sẽ sử dụng 5% mỗi nhãn cho supervised learning")
        
        unique_labels = np.unique(labels)
        print(f"- Số lượng lớp duy nhất: {len(unique_labels)}")
        print(f"- Các lớp có trong dữ liệu: {unique_labels}")
        
        # Đếm số điểm trong mỗi lớp
        for label_val in unique_labels:
            count = np.sum(labels == label_val)
            print(f"  + Lớp {label_val}: {count} điểm ({count/N*100:.2f}%)")
    
    # Run MPFCM
    print(f"\nRunning MPFCM with semi-supervised learning...")
    start_time = time.time()
    
    # Sử dụng semi_supervised_ratio=0.05 để tạo supervised/unsupervised split
    runtime, cluster_predict, cluster_predict_adj = run_MPFCM(L, c, N, data_new, params, similarity, labels, semi_supervised_ratio=0.05)
    
    # Cộng 1 vào kết quả cluster_predict để khớp với nhãn ban đầu (1-7)
    if cluster_predict is not None:
        cluster_predict += 1
        print("\nĐã chuyển đổi kết quả cluster từ 0-6 thành 1-7 để khớp với nhãn gốc")
    
    if cluster_predict_adj is not None:
        cluster_predict_adj += 1
        print("Đã chuyển đổi kết quả cluster_adj từ 0-6 thành 1-7 để khớp với nhãn gốc")
    
    end_time = time.time()
    print(f"MPFCM completed in {runtime:.2f} seconds")
    print(f"Got two sets of cluster predictions:")
    print(f"- Standard clusters (from u): {len(cluster_predict)} points")
    print(f"- Adjusted clusters (from u*(1-xi)): {len(cluster_predict_adj)} points")
    
    # Evaluate clustering results using scikit-learn metrics
    metrics = {}
    metrics_adj = {}
    
    # Create combined feature representation for evaluation
    combined_features = np.hstack([view for view in data])
    
    # Evaluate standard clustering (from u)
    print("\nEvaluating standard clustering (from u):")
    if len(cluster_predict) > 1:
        unique_clusters = np.unique(cluster_predict)
        if len(unique_clusters) > 1:
            try:
                # Unsupervised metrics
                silhouette = silhouette_score(combined_features, cluster_predict)
                metrics["silhouette"] = silhouette
                
                calinski = calinski_harabasz_score(combined_features, cluster_predict)
                metrics["calinski_harabasz"] = calinski
                
                davies = davies_bouldin_score(combined_features, cluster_predict)
                metrics["davies_bouldin"] = davies
                
                print(f"Clustering Quality:")
                print(f"  - Silhouette Score: {silhouette:.4f} (higher is better)")
                print(f"  - Calinski-Harabasz Index: {calinski:.4f} (higher is better)")
                print(f"  - Davies-Bouldin Index: {davies:.4f} (lower is better)")
            except Exception as e:
                print(f"Error calculating unsupervised metrics: {e}")
    
    # If ground truth labels are available, compute supervised metrics for both
    if labels is not None and np.any(labels > 0):
        try:
            # Filter out points with no ground truth (-1)
            valid_indices = labels > 0
            valid_labels = labels[valid_indices]
            
            # ƯU TIÊN: Tính metrics cho adjusted clustering (u*(1-xi))
            from sklearn.metrics import cohen_kappa_score
            
            # Metrics for adjusted clustering (u*(1-xi))
            valid_clusters_adj = cluster_predict_adj[valid_indices]
            ari_adj = adjusted_rand_score(valid_labels, valid_clusters_adj)
            metrics_adj["ari"] = ari_adj
            nmi_adj = normalized_mutual_info_score(valid_labels, valid_clusters_adj)
            metrics_adj["nmi"] = nmi_adj
            kappa_adj = cohen_kappa_score(valid_labels, valid_clusters_adj)
            metrics_adj["kappa"] = kappa_adj
            
            # In thông tin về số lượng điểm đánh giá
            total_valid_points = np.sum(valid_indices)
            metrics_adj["valid_points"] = int(total_valid_points)
            print(f"\nTổng số điểm có ground truth (nhãn > 0) được đánh giá: {total_valid_points}")
            
            print("\nSupervised metrics for adjusted clustering (from u*(1-xi)):")
            print(f"  - Adjusted Rand Index: {ari_adj:.4f} (higher is better)")
            print(f"  - Normalized Mutual Information: {nmi_adj:.4f} (higher is better)")
            print(f"  - Kappa Coefficient: {kappa_adj:.4f} (higher is better)")
            
            # Metrics for standard clustering (u)
            valid_clusters = cluster_predict[valid_indices]
            ari = adjusted_rand_score(valid_labels, valid_clusters)
            metrics["ari"] = ari
            nmi = normalized_mutual_info_score(valid_labels, valid_clusters)
            metrics["nmi"] = nmi
            kappa = cohen_kappa_score(valid_labels, valid_clusters)
            metrics["kappa"] = kappa
            
            # Lưu thông tin về số lượng điểm đánh giá
            metrics["valid_points"] = int(total_valid_points)
            
            print("\nSupervised metrics for standard clustering (from u):")
            print(f"  - Adjusted Rand Index: {ari:.4f} (higher is better)")
            print(f"  - Normalized Mutual Information: {nmi:.4f} (higher is better)")
            print(f"  - Kappa Coefficient: {kappa:.4f} (higher is better)")
        except Exception as e:
            print(f"Error calculating supervised metrics: {e}")
    
    # Thay đổi thứ tự trả về, ưu tiên cluster_predict_adj và metrics_adj
    return cluster_predict_adj, cluster_predict, metrics_adj, metrics

def visualize_top_left_clustering(part_shape, cluster_predictions, pixel_positions, save_dir='results'):
    """
    Visualize clustering results for top-left part and save to disk
    
    Args:
        part_shape: Shape of the part (height, width)
        cluster_predictions: Cluster predictions for each valid point
        pixel_positions: Positions of valid points in the part
        save_dir: Directory to save visualization
        
    Returns:
        cluster_image: Image with cluster assignments
    """
    height, width = part_shape
    
    # Create empty image for clusters
    cluster_image = np.zeros((height, width), dtype=int) - 1  # -1 means no cluster assigned
    
    # Assign cluster predictions to their positions
    for i, (row, col) in enumerate(pixel_positions):
        if i < len(cluster_predictions):
            cluster_image[row, col] = cluster_predictions[i]
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate colormap for visualization
    unique_clusters = np.unique(cluster_predictions)
    num_clusters = len(unique_clusters)
    
    # Create a colormap
    colors = plt.cm.jet(np.linspace(0, 1, num_clusters))
    
    # Create a colored version for visualization
    colored_image = np.zeros((height, width, 3))
    
    # Map cluster indices to colors
    for i, cluster in enumerate(unique_clusters):
        mask = cluster_image == cluster
        colored_image[mask] = colors[i, :3]
    
    # Save as image
    plt.figure(figsize=(10, 8))
    plt.imshow(colored_image)
    plt.title('Top-Left Part Clustering Results')
    plt.axis('off')
    plt.savefig(f'{save_dir}/top_left_clustering.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save cluster assignments
    np.save(f'{save_dir}/top_left_clusters.npy', cluster_image)
    
    print(f"Saved clustering visualization: {save_dir}/top_left_clustering.png")
    print(f"Saved cluster assignments: {save_dir}/top_left_clusters.npy")
    
    return cluster_image

def main():
    print("Top-Left Part Segmentation with MPFCM")
    print("="*50)
    
    # Load Augsburg data
    hs_data, sar_data, dsm_data, labels, valid_mask = load_augsburg_data()
    
    if hs_data is None or sar_data is None or dsm_data is None:
        print("Error loading data. Exiting.")
        return
    
    # Extract top-left part
    print("\nExtracting top-left part...")
    hs_part, sar_part, dsm_part, labels_part, indices = extract_top_left_part(
        hs_data, sar_data, dsm_data, labels, n_rows=2, n_cols=2
    )
    
    # Analyze classes in top-left part
    print("\nAnalyzing classes in top-left part...")
    unique_classes = np.unique(labels_part[labels_part >= 0])
    print(f"Unique classes in top-left part: {unique_classes}")
    print(f"Number of classes: {len(unique_classes)}")
    
    # Count points per class
    for cls in unique_classes:
        count = np.sum(labels_part == cls)
        print(f"Class {cls}: {count} points")
    
    # Determine number of clusters
    c = len(unique_classes) if len(unique_classes) > 0 else 7
    print(f"\nUsing {c} clusters for top-left part")
    
    # Parameters
    n_anchors = 4*c  # Number of anchors
    
    # Set parameters for MPFCM
    params = {
        "alpha": 20,
        "beta": 20/(100 * 5),
        "sigma": 5,
        "theta": 0.001,
        "omega": 0.1,
        "rho": 0.0001,
        "EPSILON": 0.0001,
    }
    
    print(f"\nParameters:")
    print(f"Number of clusters: {c}")
    print(f"Number of anchors: {n_anchors}")
    print(f"MPFCM parameters: {params}")
    
    # Prepare data for processing
    data_views, part_shape, mask_2d, labels_valid, pixel_positions = prepare_data_for_processing(
        hs_part, sar_part, dsm_part, labels_part
    )
    
    sample_size = len(data_views[0]) if data_views else 0
    print(f"Number of samples after preparation: {sample_size}")
    
    if sample_size == 0:
        print("No valid samples in top-left part, exiting...")
        return
    
    # Process top-left part
    cluster_predict_adj, cluster_predict, metrics_adj, metrics = process_top_left_part(
        data_views, labels_valid, c, n_anchors, params
    )
    
    # Visualize results - ƯU TIÊN adjusted clustering (u*(1-xi))
    print("\nVisualizing clustering results...")
    
    # Hiển thị adjusted cluster assignments (u*(1-xi))
    if cluster_predict_adj is not None and len(cluster_predict_adj) > 0:
        adj_cluster_image = visualize_top_left_clustering(
            part_shape, 
            cluster_predict_adj, 
            pixel_positions,
            save_dir='results'
        )
        print("✓ Generated adjusted clustering visualization (u*(1-xi))")
    
    # Hiển thị standard clustering (u) để so sánh
    if cluster_predict is not None and len(cluster_predict) > 0:
        std_cluster_image = visualize_top_left_clustering(
            part_shape, 
            cluster_predict, 
            pixel_positions,
            save_dir='results'
        )
        print("✓ Generated standard clustering visualization (u)")
    
    # Print final results
    print("\n" + "="*50)
    print("FINAL RESULTS FOR TOP-LEFT PART:")
    print("="*50)
    
    # ƯU TIÊN hiển thị kết quả cho u*(1-xi)
    if metrics_adj is not None:
        print("Kết quả phân cụm từ u*(1-xi):")
        if 'ari' in metrics_adj:
            print(f"  ARI: {metrics_adj['ari']:.4f}")
        if 'nmi' in metrics_adj:
            print(f"  NMI: {metrics_adj['nmi']:.4f}")
        if 'kappa' in metrics_adj:
            print(f"  Kappa: {metrics_adj['kappa']:.4f}")
    
    # Hiển thị kết quả cho u để so sánh
    if metrics is not None:
        print("\nKết quả phân cụm từ u (để so sánh):")
        if 'ari' in metrics:
            print(f"  ARI: {metrics['ari']:.4f}")
        if 'nmi' in metrics:
            print(f"  NMI: {metrics['nmi']:.4f}")
        if 'kappa' in metrics:
            print(f"  Kappa: {metrics['kappa']:.4f}")
    
    print("\nFiles generated in 'results/' directory:")
    print("- top_left_clustering.png (adjusted clustering visualization)")
    print("- top_left_clusters.npy (cluster assignments)")
    
    print("\nTop-left part segmentation completed!")

if __name__ == "__main__":
    main()
