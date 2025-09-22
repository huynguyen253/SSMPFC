#!/usr/bin/env python3
# Image Segmentation with Full Image Processing (No Resize)
import numpy as np
import os
import time
import cv2
from scipy.linalg import svd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for environments without GUI
import matplotlib.pyplot as plt
from pymatreader import read_mat
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import cohen_kappa_score
from sklearn.decomposition import PCA
from scipy import signal

# Import các module cần thiết từ project
from mpfcm import run_MPFCM, pred_cluster
from MV_DAGL import MV_DAGL
from ELMSC.augmented_data_matrix import compute_augmented_data_matrix
from normalization import normalize_multiview_data3

def load_muufl_data(filepath):
    """Load MUUFL data from .mat file and prepare for processing"""
    print(f"\nLoading MUUFL data from {filepath}...")
    data = read_mat(filepath)   
    hsi = data["hsi"]
    rgb = hsi["RGB"]
    
    # Extract HSI data
    hsi_data = hsi["Data"]  # Hyperspectral data
    print(f"Original HSI data shape: {hsi_data.shape}")
    
    # Filter HSI by removing noisy bands (4 first and 4 last)
    hsi_filtered = hsi_data[:, :, 4:-4] if hsi_data.shape[2] > 8 else hsi_data
    print(f"Filtered HSI shape: {hsi_filtered.shape}")
    
    # Extract and resize labels if available
    try:
        labels = hsi["sceneLabels"]["labels"]
        print(f"Original labels shape: {labels.shape}")
        has_labels = True
    except (KeyError, IndexError):
        print("No label information found in the data")
        has_labels = False
        labels = None
    
    return hsi_filtered, rgb, labels, has_labels

def prepare_data_for_processing(hsi_data, rgb_data, labels=None):
    """Prepare the data for processing algorithms"""
    height, width = hsi_data.shape[:2]
    
    # Reshape to 2D for processing (pixels x features)
    hsi_reshaped = hsi_data.reshape(-1, hsi_data.shape[-1])  # (height*width, bands)
    
    # Create position matrix for each pixel
    pixel_positions = np.zeros((height, width, 2), dtype=int)
    for i in range(height):
        for j in range(width):
            pixel_positions[i, j] = [i, j]
    pixel_positions_flat = pixel_positions.reshape(-1, 2)
    
    # Reshape RGB data
    if rgb_data is not None:
        # Check RGB data shape and ensure it can be properly reshaped
        print(f"RGB data shape: {rgb_data.shape}")
        # Make sure we can reshape it correctly - RGB should have shape (height, width, 3)
        if len(rgb_data.shape) == 3 and rgb_data.shape[2] == 3:
            rgb_reshaped = rgb_data.reshape(-1, 3)
        else:
            print("Warning: RGB data has unexpected shape. Using random values instead.")
            rgb_reshaped = np.random.rand(height*width, 3)  # Fallback random data
    else:
        rgb_reshaped = np.random.rand(height*width, 3)  # Fallback random data
    
    # Filter valid points if we have labels
    if labels is not None:
        labels_flat = labels.flatten()
        valid_indices = labels_flat != -1
        
        # Count and report
        total_points = len(labels_flat)
        valid_points = np.sum(valid_indices)
        filtered_points = total_points - valid_points
        
        print(f"Filtering unlabeled points...")
        print(f"Total points: {total_points}")
        print(f"Valid labeled points: {valid_points} ({valid_points/total_points*100:.1f}%)")
        print(f"Filtered unlabeled points: {filtered_points} ({filtered_points/total_points*100:.1f}%)")
        
        # Filter data
        rgb_reshaped = rgb_reshaped[valid_indices]
        hsi_reshaped = hsi_reshaped[valid_indices]
        labels_valid = labels_flat[valid_indices]
        pixel_positions_valid = pixel_positions_flat[valid_indices]
        
        # Create mask for mapping back
        mask = np.zeros(labels_flat.shape, dtype=bool)
        mask[valid_indices] = True
        mask_2d = mask.reshape(labels.shape)
    else:
        mask_2d = np.ones((height, width), dtype=bool)
        labels_valid = None
        pixel_positions_valid = pixel_positions_flat
    
    # Normalize data
    scaler_rgb = StandardScaler()
    scaler_hsi = StandardScaler()
    
    rgb_normalized = scaler_rgb.fit_transform(rgb_reshaped)
    hsi_normalized = scaler_hsi.fit_transform(hsi_reshaped)
    
    # Format data for algorithms
    data_views = [rgb_normalized, hsi_normalized]
    
    return data_views, (height, width), mask_2d, labels_valid, pixel_positions_valid

def process_full_image(data_views, labels, c, n_anchors, params, pixel_positions):
    """Process the full image"""
    L = len(data_views)  # Number of views
    N = len(data_views[0])  # Number of samples
    
    print(f"\nProcessing full image with {N} samples")
    
    # Normalize multiview data
    data = normalize_multiview_data3(data_views)
    
    # Apply PCA to reduce HSI dimensions
    print("Applying PCA to reduce HSI dimensions...")
    pca = PCA(n_components=min(20, data[1].shape[1]))
    data[1] = pca.fit_transform(data[1])
    print(f"HSI shape after PCA: {data[1].shape}")
    
    # Reduce dimensionality of data to avoid memory issues
    print("Applying additional dimensionality reduction to avoid memory issues...")
    
    # Target dimensions - adjust based on memory constraints
    target_dim = min(10, data[0].shape[1], data[1].shape[1])
    
    data_reduced = []
    for view_idx, view in enumerate(data):
        if view.shape[1] > target_dim:
            print(f"Reducing view {view_idx} from {view.shape[1]} to {target_dim} dimensions")
            pca = PCA(n_components=target_dim)
            data_reduced.append(pca.fit_transform(view))
        else:
            data_reduced.append(view)
    
    data = data_reduced  # Use reduced dimension data
    
    lamb = 1e-2
    
    # Run MV_DAGL
    print("Running MV_DAGL...")
    start_time = time.time()
    z_c, z_list = MV_DAGL(data, n_anchors, max_iter=100, alpha=0.01, tol=1e-6)
    end_time = time.time()
    print(f"MV_DAGL completed in {end_time - start_time:.2f} seconds")

    # Calculate similarity matrices and prepare data for MPFCM
    print("Calculating similarity matrices...")
    data_new = []
    similarity = []
    
    for l in range(L):
        tg = np.array(z_list[l].T @ z_list[l])
        similarity.append(tg)
        U, _, _ = svd(z_list[l].T, full_matrices=False)
        data_new.append(U[:, :n_anchors])

    # Add common representation
    U, _, _ = svd(z_c.T, full_matrices=False)
    data_new.append(U[:, :n_anchors])

    tg = np.array(z_c.T @ z_c)
    similarity.append(tg)
    
    # Run MPFCM with semi-supervised learning
    print("Running MPFCM with semi-supervised learning (5% labeled data)...")
    start_time = time.time()
    runtime, cluster_predict, cluster_predict_adj = run_MPFCM(L, c, N, data_new, params, similarity, labels, semi_supervised_ratio=0.05)
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
    # Check if we have enough data for evaluation
    if len(cluster_predict) > 1:
        # Check if we have more than one unique cluster
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
                
    # Evaluate adjusted clustering (from u*(1-xi))
    print("\nEvaluating adjusted clustering (from u*(1-xi)):")
    if len(cluster_predict_adj) > 1:
        # Check if we have more than one unique cluster
        unique_clusters = np.unique(cluster_predict_adj)
        if len(unique_clusters) > 1:
            try:
                # Unsupervised metrics
                silhouette = silhouette_score(combined_features, cluster_predict_adj)
                metrics_adj["silhouette"] = silhouette
                
                calinski = calinski_harabasz_score(combined_features, cluster_predict_adj)
                metrics_adj["calinski_harabasz"] = calinski
                
                davies = davies_bouldin_score(combined_features, cluster_predict_adj)
                metrics_adj["davies_bouldin"] = davies
                
                print(f"Adjusted Clustering Quality:")
                print(f"  - Silhouette Score: {silhouette:.4f} (higher is better)")
                print(f"  - Calinski-Harabasz Index: {calinski:.4f} (higher is better)")
                print(f"  - Davies-Bouldin Index: {davies:.4f} (lower is better)")
            except Exception as e:
                print(f"Error calculating unsupervised metrics for adjusted clustering: {e}")

    # If ground truth labels are available, compute supervised metrics for both
    if labels is not None and np.any(labels > 0):
        try:
            # Filter out points with no ground truth (-1)
            valid_indices = labels > 0
            valid_labels = labels[valid_indices]
            
            # Standard clustering metrics
            valid_clusters = cluster_predict[valid_indices]
            ari = adjusted_rand_score(valid_labels, valid_clusters)
            metrics["ari"] = ari
            nmi = normalized_mutual_info_score(valid_labels, valid_clusters)
            metrics["nmi"] = nmi
            
            # Calculate Kappa coefficient
            kappa = cohen_kappa_score(valid_labels, valid_clusters)
            metrics["kappa"] = kappa
            
            print("\nSupervised metrics for standard clustering (from u):")
            print(f"  - Adjusted Rand Index: {ari:.4f} (higher is better)")
            print(f"  - Normalized Mutual Information: {nmi:.4f} (higher is better)")
            print(f"  - Cohen's Kappa: {kappa:.4f} (higher is better)")
            
            # Adjusted clustering metrics
            valid_clusters_adj = cluster_predict_adj[valid_indices]
            ari_adj = adjusted_rand_score(valid_labels, valid_clusters_adj)
            metrics_adj["ari"] = ari_adj
            nmi_adj = normalized_mutual_info_score(valid_labels, valid_clusters_adj)
            metrics_adj["nmi"] = nmi_adj
            
            # Calculate Kappa coefficient for adjusted clustering
            kappa_adj = cohen_kappa_score(valid_labels, valid_clusters_adj)
            metrics_adj["kappa"] = kappa_adj
            
            print("\nSupervised metrics for adjusted clustering (from u*(1-xi)):")
            print(f"  - Adjusted Rand Index: {ari_adj:.4f} (higher is better)")
            print(f"  - Normalized Mutual Information: {nmi_adj:.4f} (higher is better)")
            print(f"  - Cohen's Kappa: {kappa_adj:.4f} (higher is better)")
        except Exception as e:
            print(f"Error calculating supervised metrics: {e}")

    return cluster_predict, cluster_predict_adj, metrics, metrics_adj, pixel_positions

def visualize_full_image_results(original_shape, hsi_data, rgb_data, cluster_predict, cluster_predict_adj, pixel_positions):
    """Save the results for the full image without visualization"""
    height, width = original_shape
    
    # Create empty images for the results
    full_cluster_image = np.full((height, width), -1, dtype=int)
    full_cluster_image_adj = np.full((height, width), -1, dtype=int)  # For adjusted clusters u*(1-xi)
    
    # Fill in the full image
    for i, pos in enumerate(pixel_positions):
        row, col = pos
        if 0 <= row < height and 0 <= col < width:
            full_cluster_image[row, col] = cluster_predict[i]
            full_cluster_image_adj[row, col] = cluster_predict_adj[i]
    
    # Calculate statistics for reporting
    diff_mask = full_cluster_image != full_cluster_image_adj
    percentage_diff = 100 * np.sum(diff_mask) / np.prod(diff_mask.shape)
    print(f"Difference between u and u*(1-xi) predictions: {percentage_diff:.2f}% of pixels")
    
    # Get cluster statistics
    unique_clusters, counts = np.unique(full_cluster_image[full_cluster_image >= 0], return_counts=True)
    unique_clusters_adj, counts_adj = np.unique(full_cluster_image_adj[full_cluster_image_adj >= 0], return_counts=True)
    
    print("Cluster distribution for standard clustering (u):")
    for i, (cluster, count) in enumerate(zip(unique_clusters, counts)):
        print(f"  Cluster {cluster}: {count} pixels ({count/np.sum(counts)*100:.2f}%)")
    
    print("\nCluster distribution for adjusted clustering (u*(1-xi)):")
    for i, (cluster, count) in enumerate(zip(unique_clusters_adj, counts_adj)):
        print(f"  Cluster {cluster}: {count} pixels ({count/np.sum(counts_adj)*100:.2f}%)")
    
    # Create directory for results
    os.makedirs('results', exist_ok=True)
    
    # Save basic images directly without displaying
    # Save standard clustering result - convert to float và chuẩn hóa
    full_cluster_float = full_cluster_image.astype(float)
    if full_cluster_float.max() != full_cluster_float.min():  # Tránh chia cho 0
        full_cluster_float = (full_cluster_float - full_cluster_float.min()) / (full_cluster_float.max() - full_cluster_float.min())
    plt.imsave('results/muufl_mpfcm_full_image_u.png', full_cluster_float, cmap='turbo')
    
    # Save adjusted clustering result - convert to float và chuẩn hóa
    full_cluster_adj_float = full_cluster_image_adj.astype(float)
    if full_cluster_adj_float.max() != full_cluster_adj_float.min():  # Tránh chia cho 0
        full_cluster_adj_float = (full_cluster_adj_float - full_cluster_adj_float.min()) / (full_cluster_adj_float.max() - full_cluster_adj_float.min())
    plt.imsave('results/muufl_mpfcm_full_image_u_xi.png', full_cluster_adj_float, cmap='turbo')
    
    # Save difference image
    diff_image = np.zeros((height, width, 3))
    
    # Chuẩn hóa an toàn hơn để đảm bảo giá trị trong khoảng [0,1]
    if full_cluster_image.max() > 0:  # Tránh chia cho 0
        normalized_clusters = (full_cluster_image.astype(float) - full_cluster_image.min()) / (full_cluster_image.max() - full_cluster_image.min() + 1e-10)
    else:
        normalized_clusters = np.zeros_like(full_cluster_image, dtype=float)
    
    # Đảm bảo không có giá trị nằm ngoài khoảng [0,1]
    normalized_clusters = np.clip(normalized_clusters, 0, 1)
    
    # Áp dụng mức xám cho các vùng không có sự khác biệt
    for i in range(3):
        diff_image[~diff_mask, i] = normalized_clusters[~diff_mask] * 0.7
    
    # Đánh dấu các điểm khác nhau bằng màu đỏ
    diff_image[diff_mask] = [1, 0, 0]  # Red for pixels that differ
    
    # Đảm bảo tất cả các giá trị đều nằm trong khoảng [0,1]
    diff_image = np.clip(diff_image, 0, 1)
    
    # Lưu hình ảnh
    plt.imsave('results/muufl_mpfcm_full_image_difference.png', diff_image)
    
    # Save RGB reference if available
    if rgb_data is not None:
        rgb_vis = rgb_data / 255.0 if rgb_data.max() > 1 else rgb_data
        plt.imsave('results/muufl_mpfcm_full_image_rgb.png', rgb_vis)
        
    # Print saved file information
    print(f"Images saved to:")
    print(f"- results/muufl_mpfcm_full_image_u.png (standard clustering)")
    print(f"- results/muufl_mpfcm_full_image_u_xi.png (adjusted clustering)")
    print(f"- results/muufl_mpfcm_full_image_difference.png (difference visualization)")
    
    # Save segmentation results
    np.save('results/muufl_mpfcm_labels_full_image.npy', full_cluster_image)
    np.save('results/muufl_mpfcm_labels_full_image_adj.npy', full_cluster_image_adj)
    
    # Also save metrics to a JSON file
    import json
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    # Get cluster statistics
    cluster_stats = {
        "standard": {
            "unique_clusters": len(unique_clusters),
            "cluster_counts": dict(zip([int(c) for c in unique_clusters], [int(cnt) for cnt in counts]))
        },
        "adjusted": {
            "unique_clusters": len(unique_clusters_adj),
            "cluster_counts": dict(zip([int(c) for c in unique_clusters_adj], [int(cnt) for cnt in counts_adj]))
        },
        "difference_percentage": percentage_diff
    }
    
    with open('results/muufl_mpfcm_full_image_metrics.json', 'w') as f:
        json.dump(cluster_stats, f, default=convert_to_json_serializable, indent=4)
    
    return full_cluster_image, full_cluster_image_adj

def main():
    print("Starting MUUFL Full Image Segmentation using MPFCM with MV_DAGL...")
    
    # Load MUUFL data
    filepath = "data/muufl_gulfport_campus_1_hsi_220_label.mat"
    hsi_data, rgb_data, labels, has_labels = load_muufl_data(filepath)
    
    if hsi_data is None:
        print("Error loading data. Exiting.")
        return
    
    # Get original dimensions
    height, width = hsi_data.shape[:2]
    print(f"Full image dimensions: {height}x{width}")
    
    # Create output directories
    os.makedirs('results', exist_ok=True)
    
    # Parameters
    c = 11  # Number of clusters
    n_anchors = c  # Number of anchors
    
    # Set parameters for MPFCM
    alpha = 20
    beta = alpha/(100 * 4)
    sigma = 5
    omega = 1
    theta = 0.001
    rho = 1

    params = {
        "alpha": alpha,
        "beta": beta,
        "sigma": sigma,
        "theta": theta,
        "omega": omega,
        "rho": rho
    }
    
    print(f"\nParameters:")
    print(f"Number of clusters: {c}")
    print(f"Number of anchors: {n_anchors}")
    print(f"MPFCM parameters: {params}")
    
    # Prepare data for processing
    data_views, image_shape, mask_2d, labels_valid, pixel_positions = prepare_data_for_processing(
        hsi_data, rgb_data, labels
    )
    
    # Process full image
    cluster_predict, cluster_predict_adj, metrics, metrics_adj, pixel_positions = process_full_image(
        data_views, labels_valid, c, n_anchors, params, pixel_positions
    )
    
    # Visualize and save the results
    full_cluster_image, full_cluster_image_adj = visualize_full_image_results(
        (height, width), hsi_data, rgb_data, cluster_predict, cluster_predict_adj, pixel_positions
    )
    
    print(f"\nFull image segmentation completed!")
    print(f"Results saved to:")
    print(f"- results/muufl_mpfcm_full_image_comparison.png (comparison visualization)")
    print(f"- results/muufl_mpfcm_labels_full_image.npy (regular clustering)")
    print(f"- results/muufl_mpfcm_labels_full_image_adj.npy (adjusted clustering)")
    print(f"- results/muufl_mpfcm_full_image_metrics.json (metrics)")

if __name__ == "__main__":
    main()