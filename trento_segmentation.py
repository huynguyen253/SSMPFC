# Trento Image Segmentation with MPFCM
import numpy as np
import os
import scipy.io as sio
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time
import json
from sklearn.preprocessing import StandardScaler
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
        data = loadmat(filepath)
        return data
    except Exception as e:
        try:
            f = h5py.File(filepath, 'r')
            data = {}
            for k, v in f.items():
                data[k] = np.array(v)
            return data
        except Exception as e2:
            print(f"Error loading MAT file: {e2}")
            return None

def load_trento_data(data_dir="D:/mpfcm_8_6_12 (1)/mpfcm_8_6_12/data/Trento"):
    """Load the Trento dataset."""
    print(f"\nLoading Trento data from {data_dir}...")
    
    # File paths
    gt_file = os.path.join(data_dir, "GT_Trento.mat")
    hsi_file = os.path.join(data_dir, "HSI_Trento.mat")
    lidar_file = os.path.join(data_dir, "Lidar_Trento.mat")
    
    # Load Ground Truth data
    print(f"Loading Ground Truth data...")
    gt_data = read_mat(gt_file)
    if gt_data is None:
        return None, None, None, None
    
    gt_key = [key for key in gt_data.keys() if not key.startswith('__')][0]
    labels = gt_data[gt_key]
    
    # QUAN TRỌNG: Chuyển đổi labels từ uint8 sang int32 để tránh overflow
    labels = labels.astype(np.int32)
    
    print(f"Ground Truth shape: {labels.shape}")
    print(f"Labels dtype: {labels.dtype}")
    
    # Load HSI data
    print(f"Loading HSI data...")
    hsi_data = read_mat(hsi_file)
    if hsi_data is None:
        return None, None, None, None
    
    hsi_key = [key for key in hsi_data.keys() if not key.startswith('__')][0]
    hsi = hsi_data[hsi_key]
    print(f"HSI data shape: {hsi.shape}")
    
    # Load Lidar data
    print(f"Loading Lidar data...")
    lidar_data = read_mat(lidar_file)
    if lidar_data is None:
        return None, None, None, None
    
    lidar_key = [key for key in lidar_data.keys() if not key.startswith('__')][0]
    lidar = lidar_data[lidar_key]
    print(f"Lidar data shape: {lidar.shape}")
    
    # Create mask for valid points (nhãn > 0 là valid, nhãn = 0 là không có ground truth)
    valid_mask = (labels > 0)
    
    # Count valid points
    valid_points = np.sum(valid_mask)
    total_pixels = labels.size
    print(f"Total pixels: {total_pixels}")
    print(f"Valid labeled points (label > 0): {valid_points} ({valid_points/total_pixels*100:.1f}%)")
    
    # Print statistics about classes
    unique_classes = np.unique(labels[valid_mask])
    print(f"Unique classes: {unique_classes}")
    print(f"Number of classes: {len(unique_classes)}")
    
    # Print label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Label distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  Label {label}: {count} samples ({count/total_pixels*100:.1f}%)")
    
    return hsi, lidar, labels, valid_mask

def prepare_data_for_processing(hsi_data, lidar_data, labels):
    """Prepare the data for processing algorithms."""
    height, width = hsi_data.shape[:2]
    
    # Reshape to 2D for processing
    hsi_reshaped = hsi_data.reshape(-1, hsi_data.shape[-1])
    lidar_reshaped = lidar_data.reshape(-1, 1)
    
    # Create position matrix for each pixel
    pixel_positions = np.zeros((height, width, 2), dtype=int)
    for i in range(height):
        for j in range(width):
            pixel_positions[i, j] = [i, j]
    pixel_positions_flat = pixel_positions.reshape(-1, 2)
    
    # Filter valid points (chỉ lấy điểm có nhãn > 0)
    if labels is not None:
        labels_flat = labels.flatten()
        valid_indices = labels_flat > 0  # Lọc điểm có nhãn > 0
        
        # Count and report
        total_points = len(labels_flat)
        valid_points = np.sum(valid_indices)
        filtered_points = total_points - valid_points
        
        print(f"Filtering unlabeled points...")
        print(f"Total points: {total_points}")
        print(f"Valid labeled points: {valid_points} ({valid_points/total_points*100:.1f}%)")
        print(f"Filtered unlabeled points: {filtered_points} ({filtered_points/total_points*100:.1f}%)")
        
        # Filter data
        hsi_reshaped = hsi_reshaped[valid_indices]
        lidar_reshaped = lidar_reshaped[valid_indices]
        labels_valid = labels_flat[valid_indices]
        pixel_positions_valid = pixel_positions_flat[valid_indices]
        
        # Print distribution of valid labels
        unique_valid_labels, valid_counts = np.unique(labels_valid, return_counts=True)
        print("Valid label distribution:")
        for label, count in zip(unique_valid_labels, valid_counts):
            print(f"  Label {label}: {count} samples ({count/len(labels_valid)*100:.1f}%)")
        
        # Create mask for mapping back
        mask = np.zeros(labels_flat.shape, dtype=bool)
        mask[valid_indices] = True
        mask_2d = mask.reshape(labels.shape)
    else:
        mask_2d = np.ones((height, width), dtype=bool)
        labels_valid = None
        pixel_positions_valid = pixel_positions_flat
    
    # Normalize data
    scaler_hsi = StandardScaler()
    scaler_lidar = StandardScaler()
    
    hsi_normalized = scaler_hsi.fit_transform(hsi_reshaped)
    lidar_normalized = scaler_lidar.fit_transform(lidar_reshaped)
    
    # Format data for algorithms (2 views: HSI, Lidar)
    data_views = [hsi_normalized, lidar_normalized]
    
    return data_views, (height, width), mask_2d, labels_valid, pixel_positions_valid

def process_trento_data(data_views, labels, c, n_anchors, params):
    """Process Trento data with MPFCM"""
    L = len(data_views)  # Number of views (HSI, Lidar)
    N = len(data_views[0])  # Number of samples
    
    print(f"\nProcessing Trento data with {N} samples and {L} views")
    
    # Normalize multiview data
    data = normalize_multiview_data3(data_views)
    
    # Sử dụng MV_DAGL để tạo biểu diễn chung và riêng
    print("Running MV-DAGL...")
    start_time = time.time()
    Z_common, Z_specific = MV_DAGL(data, n_anchors, max_iter=100, alpha=0.2, tol=1e-6)
    end_time = time.time()
    print(f"MV-DAGL completed in {end_time - start_time:.2f} seconds")
    
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
    
    # Run MPFCM
    print(f"\nRunning MPFCM with semi-supervised learning...")
    start_time = time.time()
    
    # Sử dụng semi_supervised_ratio=0.05 để tạo supervised/unsupervised split
    runtime, cluster_predict, cluster_predict_adj = run_MPFCM(L, c, N, data_new, params, similarity, labels, semi_supervised_ratio=0.05)
    
    # Cộng 1 vào kết quả cluster_predict để khớp với nhãn ban đầu (1-6)
    if cluster_predict is not None:
        cluster_predict += 1
        print("\nĐã chuyển đổi kết quả cluster từ 0-5 thành 1-6 để khớp với nhãn gốc")
    
    if cluster_predict_adj is not None:
        cluster_predict_adj += 1
        print("Đã chuyển đổi kết quả cluster_adj từ 0-5 thành 1-6 để khớp với nhãn gốc")
    
    end_time = time.time()
    print(f"MPFCM completed in {runtime:.2f} seconds")
    
    # Evaluate clustering results
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
                print(f"  - Silhouette Score: {silhouette:.4f}")
                print(f"  - Calinski-Harabasz Index: {calinski:.4f}")
                print(f"  - Davies-Bouldin Index: {davies:.4f}")
            except Exception as e:
                print(f"Error calculating unsupervised metrics: {e}")
    
    # If ground truth labels are available, compute supervised metrics
    if labels is not None and np.any(labels > 0):
        try:
            from sklearn.metrics import cohen_kappa_score, adjusted_rand_score, normalized_mutual_info_score
            
            # Filter out points with no ground truth (0)
            valid_indices = labels > 0
            valid_labels = labels[valid_indices]
            
            # Metrics for adjusted clustering (u*(1-xi))
            if cluster_predict_adj is not None:
                valid_clusters_adj = cluster_predict_adj[valid_indices]
                ari_adj = adjusted_rand_score(valid_labels, valid_clusters_adj)
                metrics_adj["ari"] = ari_adj
                nmi_adj = normalized_mutual_info_score(valid_labels, valid_clusters_adj)
                metrics_adj["nmi"] = nmi_adj
                kappa_adj = cohen_kappa_score(valid_labels, valid_clusters_adj)
                metrics_adj["kappa"] = kappa_adj
                
                print("\nSupervised metrics for adjusted clustering (from u*(1-xi)):")
                print(f"  - Adjusted Rand Index: {ari_adj:.4f}")
                print(f"  - Normalized Mutual Information: {nmi_adj:.4f}")
                print(f"  - Kappa Coefficient: {kappa_adj:.4f}")
            
            # Metrics for standard clustering (u)
            valid_clusters = cluster_predict[valid_indices]
            ari = adjusted_rand_score(valid_labels, valid_clusters)
            metrics["ari"] = ari
            nmi = normalized_mutual_info_score(valid_labels, valid_clusters)
            metrics["nmi"] = nmi
            kappa = cohen_kappa_score(valid_labels, valid_clusters)
            metrics["kappa"] = kappa
            
            print("\nSupervised metrics for standard clustering (from u):")
            print(f"  - Adjusted Rand Index: {ari:.4f}")
            print(f"  - Normalized Mutual Information: {nmi:.4f}")
            print(f"  - Kappa Coefficient: {kappa:.4f}")
        except Exception as e:
            print(f"Error calculating supervised metrics: {e}")
    
    return cluster_predict_adj, cluster_predict, metrics_adj, metrics

def visualize_clustering_results(hsi_data, lidar_data, labels, cluster_predict, cluster_predict_adj, 
                                pixel_positions, mask_2d, save_dir='results'):
    """Visualize clustering results for Trento data"""
    height, width = hsi_data.shape[:2]
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    
    # Create empty images for cluster assignments
    cluster_image = np.zeros((height, width), dtype=int) - 1
    cluster_image_adj = np.zeros((height, width), dtype=int) - 1
    
    # Assign cluster predictions to their positions
    for i, (row, col) in enumerate(pixel_positions):
        if i < len(cluster_predict):
            cluster_image[row, col] = cluster_predict[i]
        if cluster_predict_adj is not None and i < len(cluster_predict_adj):
            cluster_image_adj[row, col] = cluster_predict_adj[i]
    
    # Save cluster assignments
    np.save(f'{save_dir}/trento_cluster_labels.npy', cluster_image)
    if cluster_predict_adj is not None:
        np.save(f'{save_dir}/trento_cluster_labels_adj.npy', cluster_image_adj)
    
    # Generate colormap for visualization
    unique_clusters = np.unique(cluster_predict)
    num_clusters = len(unique_clusters)
    colors = plt.cm.jet(np.linspace(0, 1, num_clusters))
    
    # Create colored versions for visualization
    colored_image = np.zeros((height, width, 3))
    colored_image_adj = np.zeros((height, width, 3))
    
    # Map cluster indices to colors
    for i, cluster in enumerate(unique_clusters):
        mask = cluster_image == cluster
        colored_image[mask] = colors[i, :3]
        
        if cluster_predict_adj is not None:
            mask_adj = cluster_image_adj == cluster
            colored_image_adj[mask_adj] = colors[i, :3]
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original HSI visualization (first 3 bands)
    if hsi_data.shape[2] >= 3:
        hsi_vis = hsi_data[:, :, :3]
        hsi_vis = (hsi_vis - np.min(hsi_vis)) / (np.max(hsi_vis) - np.min(hsi_vis))
        axes[0, 0].imshow(hsi_vis)
        axes[0, 0].set_title('HSI Data (RGB)')
        axes[0, 0].axis('off')
    
    # Lidar visualization
    axes[0, 1].imshow(lidar_data, cmap='gray')
    axes[0, 1].set_title('Lidar Data')
    axes[0, 1].axis('off')
    
    # Ground truth
    gt_vis = labels.copy()
    gt_vis[labels == 0] = np.nan
    axes[0, 2].imshow(gt_vis, cmap='tab10')
    axes[0, 2].set_title('Ground Truth')
    axes[0, 2].axis('off')
    
    # Standard clustering
    axes[1, 0].imshow(colored_image)
    axes[1, 0].set_title('Standard Clustering (u)')
    axes[1, 0].axis('off')
    
    # Adjusted clustering
    if cluster_predict_adj is not None:
        axes[1, 1].imshow(colored_image_adj)
        axes[1, 1].set_title('Adjusted Clustering (u*(1-xi))')
        axes[1, 1].axis('off')
    else:
        axes[1, 1].text(0.5, 0.5, 'No adjusted clustering', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Adjusted Clustering (u*(1-xi))')
        axes[1, 1].axis('off')
    
    # Valid points mask
    axes[1, 2].imshow(mask_2d, cmap='gray')
    axes[1, 2].set_title('Valid Points Mask')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/trento_clustering_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return cluster_image, cluster_image_adj

def main():
    print("Starting Trento Image Segmentation using MPFCM with MV_DAGL...")
    
    # Load Trento data
    hsi_data, lidar_data, labels, valid_mask = load_trento_data()
    
    if hsi_data is None or lidar_data is None or labels is None:
        print("Error loading data. Exiting.")
        return
    
    # Get original dimensions
    height, width = hsi_data.shape[:2]
    print(f"Original image dimensions: {height}x{width}")
    
    # Determine number of clusters based on unique classes
    unique_classes = np.unique(labels[labels > 0])
    c = len(unique_classes)
    print(f"Number of unique classes: {c}")
    print(f"Classes: {unique_classes}")
    
    # Parameters
    n_anchors = 4 * c  # Number of anchors
    
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
    data_views, image_shape, mask_2d, labels_valid, pixel_positions = prepare_data_for_processing(
        hsi_data, lidar_data, labels
    )
    
    sample_size = len(data_views[0]) if data_views else 0
    print(f"Number of samples after preparation: {sample_size}")
    
    if sample_size == 0:
        print("No valid samples, exiting...")
        return
    
    # Process data with MPFCM
    cluster_predict_adj, cluster_predict, metrics_adj, metrics = process_trento_data(
        data_views, labels_valid, c, n_anchors, params
    )
    
    # Visualize results
    cluster_image, cluster_image_adj = visualize_clustering_results(
        hsi_data, lidar_data, labels, cluster_predict, cluster_predict_adj,
        pixel_positions, mask_2d
    )
    
    # Print final summary
    print("\n" + "="*80)
    print("KẾT QUẢ CUỐI CÙNG:")
    print("="*80)
    
    # ƯU TIÊN hiển thị kết quả cho u*(1-xi)
    if metrics_adj:
        print("Kết quả phân cụm từ u*(1-xi):")
        print(f"ARI: {metrics_adj['ari']:.4f}")
        print(f"NMI: {metrics_adj['nmi']:.4f}")
        print(f"Kappa: {metrics_adj['kappa']:.4f}")
    
    # Hiển thị kết quả cho u để so sánh
    if metrics:
        print("\nKết quả phân cụm từ u (để so sánh):")
        print(f"ARI: {metrics['ari']:.4f}")
        print(f"NMI: {metrics['nmi']:.4f}")
        print(f"Kappa: {metrics['kappa']:.4f}")
    
    print(f"\nResults saved to:")
    print(f"- results/trento_clustering_results.png")
    print(f"- results/trento_cluster_labels.npy")
    print("="*80)

if __name__ == "__main__":
    main() 