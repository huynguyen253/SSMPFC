# Trento Image Segmentation with MPFCM - Divided Processing
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

def save_part_data(part_name, hsi_data, lidar_data, labels, indices):
    """Save part data to files"""
    save_dir = "parts_trento"
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(f'{save_dir}/{part_name}_hsi.npy', hsi_data)
    np.save(f'{save_dir}/{part_name}_lidar.npy', lidar_data)
    np.save(f'{save_dir}/{part_name}_labels.npy', labels)
    np.save(f'{save_dir}/{part_name}_indices.npy', indices)
    
    # Save metadata
    metadata = {
        'part_name': part_name,
        'hsi_shape': hsi_data.shape,
        'lidar_shape': lidar_data.shape,
        'labels_shape': labels.shape,
        'num_valid_points': len(labels),
        'unique_labels': np.unique(labels).tolist()
    }
    
    with open(f'{save_dir}/{part_name}_metadata.txt', 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

def load_part_data(part_name):
    """Load part data from files"""
    save_dir = "parts_trento"
    
    hsi_data = np.load(f'{save_dir}/{part_name}_hsi.npy')
    lidar_data = np.load(f'{save_dir}/{part_name}_lidar.npy')
    labels = np.load(f'{save_dir}/{part_name}_labels.npy')
    indices = np.load(f'{save_dir}/{part_name}_indices.npy')
    
    return hsi_data, lidar_data, labels, indices

def visualize_part_clustering(part_name, image_shape, cluster_predictions, pixel_positions, save_dir='parts_trento_clusters'):
    """Visualize clustering results for a specific part"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create empty image for cluster assignments
    cluster_image = np.zeros(image_shape, dtype=int) - 1
    
    # Assign cluster predictions to their positions
    # pixel_positions chứa tọa độ tương đối trong phần
    for i, (row, col) in enumerate(pixel_positions):
        if i < len(cluster_predictions):
            cluster_image[row, col] = cluster_predictions[i]
    
    # Save cluster assignments
    np.save(f'{save_dir}/{part_name}_clusters.npy', cluster_image)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(cluster_image, cmap='tab10')
    plt.title(f'Clustering Results - {part_name}')
    plt.colorbar()
    plt.axis('off')
    plt.savefig(f'{save_dir}/{part_name}_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return cluster_image

def visualize_part_results(full_shape, hsi_data, lidar_data, parts_results, n_rows, n_cols):
    """Visualize all parts together"""
    height, width = full_shape[:2]
    
    # Create figure for all parts
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 16))
    axes = axes.flatten()
    
    for i, (part_name, part_data) in enumerate(parts_results.items()):
        if i < len(axes):
            cluster_image = part_data['cluster_image']
            axes[i].imshow(cluster_image, cmap='tab10')
            axes[i].set_title(f'{part_name}')
            axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(len(parts_results), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/trento_parts_clustering.png', dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_overall_clustering(hsi_data, ground_truth, parts_results, n_rows, n_cols, valid_mask=None):
    """Evaluate overall clustering performance"""
    height, width = hsi_data.shape[:2]
    
    # Create combined cluster image
    combined_clusters = np.zeros((height, width), dtype=int) - 1
    
    for part_name, part_data in parts_results.items():
        cluster_image = part_data['cluster_image']
        start_h = part_data['start_h']
        end_h = part_data['end_h']
        start_w = part_data['start_w']
        end_w = part_data['end_w']
        
        # Assign cluster results to combined image
        # cluster_image có kích thước của phần, cần cắt cho phù hợp với vùng được chỉ định
        part_h, part_w = cluster_image.shape
        combined_clusters[start_h:start_h+part_h, start_w:start_w+part_w] = cluster_image
    
    # Evaluate metrics
    metrics = {}
    
    if valid_mask is not None:
        valid_indices = valid_mask.flatten()
        valid_clusters = combined_clusters.flatten()[valid_indices]
        valid_labels = ground_truth.flatten()[valid_indices]
        
        # Remove invalid points (-1)
        valid_mask_clean = valid_clusters != -1
        if np.any(valid_mask_clean):
            valid_clusters_clean = valid_clusters[valid_mask_clean]
            valid_labels_clean = valid_labels[valid_mask_clean]
            
            try:
                from sklearn.metrics import cohen_kappa_score, adjusted_rand_score, normalized_mutual_info_score
                
                ari = adjusted_rand_score(valid_labels_clean, valid_clusters_clean)
                nmi = normalized_mutual_info_score(valid_labels_clean, valid_clusters_clean)
                kappa = cohen_kappa_score(valid_labels_clean, valid_clusters_clean)
                
                metrics = {
                    'ari': ari,
                    'nmi': nmi,
                    'kappa': kappa
                }
                
                print(f"\nOverall Clustering Metrics:")
                print(f"ARI: {ari:.4f}")
                print(f"NMI: {nmi:.4f}")
                print(f"Kappa: {kappa:.4f}")
                
            except Exception as e:
                print(f"Error calculating metrics: {e}")
    
    # Save combined results
    np.save('results/trento_combined_clusters.npy', combined_clusters)
    
    # Visualize combined results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    if hsi_data.shape[2] >= 3:
        hsi_vis = hsi_data[:, :, :3]
        hsi_vis = (hsi_vis - np.min(hsi_vis)) / (np.max(hsi_vis) - np.min(hsi_vis))
        plt.imshow(hsi_vis)
        plt.title('HSI Data (RGB)')
        plt.axis('off')
    
    plt.subplot(1, 3, 2)
    gt_vis = ground_truth.copy()
    gt_vis[ground_truth == 0] = np.nan
    plt.imshow(gt_vis, cmap='tab10')
    plt.title('Ground Truth')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(combined_clusters, cmap='tab10')
    plt.title('Combined Clustering Results')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/trento_combined_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return combined_clusters, metrics

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

def divide_into_parts(image, labels=None, n_rows=1, n_cols=2):
    """Divide image into parts"""
    height, width = image.shape[:2]
    
    parts = {}
    part_height = height // n_rows
    part_width = width // n_cols
    
    for i in range(n_rows):
        for j in range(n_cols):
            part_name = f"part_{i}_{j}"
            
            # Calculate boundaries
            start_h = i * part_height
            end_h = start_h + part_height if i < n_rows - 1 else height
            start_w = j * part_width
            end_w = start_w + part_width if j < n_cols - 1 else width
            
            # Extract part
            if len(image.shape) == 3:
                part_image = image[start_h:end_h, start_w:end_w, :]
            else:
                part_image = image[start_h:end_h, start_w:end_w]
            
            parts[part_name] = {
                'image': part_image,
                'start_h': start_h,
                'end_h': end_h,
                'start_w': start_w,
                'end_w': end_w
            }
            
            # Extract labels if provided
            if labels is not None:
                part_labels = labels[start_h:end_h, start_w:end_w]
                parts[part_name]['labels'] = part_labels
    
    return parts

def prepare_data_for_processing(hsi_part, lidar_part, labels_part=None, start_h=None, end_h=None, start_w=None, end_w=None):
    """Prepare part data for processing algorithms."""
    height, width = hsi_part.shape[:2]
    
    # Reshape to 2D for processing
    hsi_reshaped = hsi_part.reshape(-1, hsi_part.shape[-1])
    lidar_reshaped = lidar_part.reshape(-1, 1)
    
    # Create position matrix for each pixel
    pixel_positions = np.zeros((height, width, 2), dtype=int)
    for i in range(height):
        for j in range(width):
            pixel_positions[i, j] = [i, j]  # Local coordinates within the part
    pixel_positions_flat = pixel_positions.reshape(-1, 2)
    
    # Filter valid points (chỉ lấy điểm có nhãn > 0)
    if labels_part is not None:
        labels_flat = labels_part.flatten()
        valid_indices = labels_flat > 0  # Lọc điểm có nhãn > 0
        
        # Count and report
        total_points = len(labels_flat)
        valid_points = np.sum(valid_indices)
        filtered_points = total_points - valid_points
        
        print(f"  Filtering unlabeled points...")
        print(f"  Total points: {total_points}")
        print(f"  Valid labeled points: {valid_points} ({valid_points/total_points*100:.1f}%)")
        print(f"  Filtered unlabeled points: {filtered_points} ({filtered_points/total_points*100:.1f}%)")
        
        # Filter data
        hsi_reshaped = hsi_reshaped[valid_indices]
        lidar_reshaped = lidar_reshaped[valid_indices]
        labels_valid = labels_flat[valid_indices]
        pixel_positions_valid = pixel_positions_flat[valid_indices]
        
        # Print distribution of valid labels
        unique_valid_labels, valid_counts = np.unique(labels_valid, return_counts=True)
        print("  Valid label distribution:")
        for label, count in zip(unique_valid_labels, valid_counts):
            print(f"    Label {label}: {count} samples ({count/len(labels_valid)*100:.1f}%)")
        
        # Create mask for mapping back
        mask = np.zeros(labels_flat.shape, dtype=bool)
        mask[valid_indices] = True
        mask_2d = mask.reshape(labels_part.shape)
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

def process_part(data_views, labels, c, n_anchors, params):
    """Process a single part with MPFCM"""
    L = len(data_views)  # Number of views (HSI, Lidar)
    N = len(data_views[0])  # Number of samples
    
    print(f"  Processing part with {N} samples and {L} views")
    
    # Normalize multiview data
    data = normalize_multiview_data3(data_views)
    
    # Sử dụng MV_DAGL để tạo biểu diễn chung và riêng
    print("  Running MV-DAGL...")
    start_time = time.time()
    Z_common, Z_specific = MV_DAGL(data, n_anchors, max_iter=100, alpha=0.2, tol=1e-6)
    end_time = time.time()
    print(f"  MV-DAGL completed in {end_time - start_time:.2f} seconds")
    
    # Calculate similarity matrices and prepare data for MPFCM
    print("  Calculating similarity matrices...")
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
    print(f"  Running MPFCM with semi-supervised learning...")
    start_time = time.time()
    
    # Sử dụng semi_supervised_ratio=0.05 để tạo supervised/unsupervised split
    runtime, cluster_predict, cluster_predict_adj = run_MPFCM(L, c, N, data_new, params, similarity, labels, semi_supervised_ratio=0.05)
    
    # Cộng 1 vào kết quả cluster_predict để khớp với nhãn ban đầu (1-6)
    if cluster_predict is not None:
        cluster_predict += 1
        print("  Đã chuyển đổi kết quả cluster từ 0-5 thành 1-6 để khớp với nhãn gốc")
    
    if cluster_predict_adj is not None:
        cluster_predict_adj += 1
        print("  Đã chuyển đổi kết quả cluster_adj từ 0-5 thành 1-6 để khớp với nhãn gốc")
    
    end_time = time.time()
    print(f"  MPFCM completed in {runtime:.2f} seconds")
    
    # Evaluate clustering results
    metrics = {}
    metrics_adj = {}
    
    # Create combined feature representation for evaluation
    combined_features = np.hstack([view for view in data])
    
    # Evaluate standard clustering (from u)
    print("  Evaluating standard clustering (from u):")
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
                
                print(f"  Clustering Quality:")
                print(f"    - Silhouette Score: {silhouette:.4f}")
                print(f"    - Calinski-Harabasz Index: {calinski:.4f}")
                print(f"    - Davies-Bouldin Index: {davies:.4f}")
            except Exception as e:
                print(f"  Error calculating unsupervised metrics: {e}")
    
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
                
                print("  Supervised metrics for adjusted clustering (from u*(1-xi)):")
                print(f"    - Adjusted Rand Index: {ari_adj:.4f}")
                print(f"    - Normalized Mutual Information: {nmi_adj:.4f}")
                print(f"    - Kappa Coefficient: {kappa_adj:.4f}")
            
            # Metrics for standard clustering (u)
            valid_clusters = cluster_predict[valid_indices]
            ari = adjusted_rand_score(valid_labels, valid_clusters)
            metrics["ari"] = ari
            nmi = normalized_mutual_info_score(valid_labels, valid_clusters)
            metrics["nmi"] = nmi
            kappa = cohen_kappa_score(valid_labels, valid_clusters)
            metrics["kappa"] = kappa
            
            print("  Supervised metrics for standard clustering (from u):")
            print(f"    - Adjusted Rand Index: {ari:.4f}")
            print(f"    - Normalized Mutual Information: {nmi:.4f}")
            print(f"    - Kappa Coefficient: {kappa:.4f}")
        except Exception as e:
            print(f"  Error calculating supervised metrics: {e}")
    
    return cluster_predict_adj, cluster_predict, metrics_adj, metrics

def main(force_reload=True, n_rows=1, n_cols=2):
    print("Starting Trento Image Segmentation using MPFCM with MV_DAGL (Divided Processing)...")
    
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
    print(f"Image division: {n_rows}x{n_cols} parts")
    
    # Divide image into parts
    print(f"\nDividing image into {n_rows}x{n_cols} parts...")
    hsi_parts = divide_into_parts(hsi_data, labels, n_rows, n_cols)
    lidar_parts = divide_into_parts(lidar_data, None, n_rows, n_cols)
    
    # Process each part
    parts_results = {}
    parts_metrics = {}
    parts_metrics_adj = {}
    
    for part_name in hsi_parts.keys():
        print(f"\n{'='*60}")
        print(f"Processing {part_name}...")
        print(f"{'='*60}")
        
        # Get part data
        hsi_part = hsi_parts[part_name]['image']
        lidar_part = lidar_parts[part_name]['image']
        labels_part = hsi_parts[part_name]['labels']
        start_h = hsi_parts[part_name]['start_h']
        end_h = hsi_parts[part_name]['end_h']
        start_w = hsi_parts[part_name]['start_w']
        end_w = hsi_parts[part_name]['end_w']
        
        print(f"Part dimensions: {hsi_part.shape}")
        print(f"Part boundaries: ({start_h}:{end_h}, {start_w}:{end_w})")
        
        # Check if part has enough valid points
        valid_points_in_part = np.sum(labels_part > 0)
        if valid_points_in_part < 10:
            print(f"  Skipping {part_name} - insufficient valid points ({valid_points_in_part})")
            continue
        
        # Prepare data for processing
        data_views, image_shape, mask_2d, labels_valid, pixel_positions = prepare_data_for_processing(
            hsi_part, lidar_part, labels_part, start_h, end_h, start_w, end_w
        )
        
        sample_size = len(data_views[0]) if data_views else 0
        print(f"  Number of samples after preparation: {sample_size}")
        
        if sample_size == 0:
            print(f"  Skipping {part_name} - no valid samples")
            continue
        
        # Process part with MPFCM
        cluster_predict_adj, cluster_predict, metrics_adj, metrics = process_part(
            data_views, labels_valid, c, n_anchors, params
        )
        
        # Store results
        parts_metrics[part_name] = metrics
        parts_metrics_adj[part_name] = metrics_adj
        
        # Visualize part results
        if cluster_predict is not None:
            cluster_image = visualize_part_clustering(
                part_name, image_shape, cluster_predict, pixel_positions
            )
            # Store cluster image with part boundaries for later combination
            parts_results[part_name] = {
                'cluster_image': cluster_image,
                'start_h': start_h,
                'end_h': end_h,
                'start_w': start_w,
                'end_w': end_w
            }
        
        # Save part data
        save_part_data(part_name, hsi_part, lidar_part, labels_part, pixel_positions)
    
    # Visualize all parts together
    if parts_results:
        print(f"\nVisualizing all parts together...")
        visualize_part_results(
            (height, width), hsi_data, lidar_data, parts_results, n_rows, n_cols
        )
        
        # Evaluate overall clustering
        print(f"\nEvaluating overall clustering...")
        combined_clusters, overall_metrics = evaluate_overall_clustering(
            hsi_data, labels, parts_results, n_rows, n_cols, valid_mask
        )
        
        # Print final summary
        print("\n" + "="*80)
        print("KẾT QUẢ CUỐI CÙNG:")
        print("="*80)
        
        # Calculate average metrics across parts
        if parts_metrics_adj:
            valid_metrics = [m for m in parts_metrics_adj.values() if m and 'ari' in m]
            if valid_metrics:
                avg_ari = np.mean([m.get('ari', 0) for m in valid_metrics])
                avg_nmi = np.mean([m.get('nmi', 0) for m in valid_metrics])
                avg_kappa = np.mean([m.get('kappa', 0) for m in valid_metrics])
                
                print("Kết quả trung bình từ các phần (u*(1-xi)):")
                print(f"ARI: {avg_ari:.4f}")
                print(f"NMI: {avg_nmi:.4f}")
                print(f"Kappa: {avg_kappa:.4f}")
            else:
                print("Không có metrics hợp lệ để tính trung bình")
        
        if overall_metrics:
            print("\nKết quả tổng hợp:")
            print(f"ARI: {overall_metrics.get('ari', 0):.4f}")
            print(f"NMI: {overall_metrics.get('nmi', 0):.4f}")
            print(f"Kappa: {overall_metrics.get('kappa', 0):.4f}")
        
        print(f"\nResults saved to:")
        print(f"- results/trento_parts_clustering.png")
        print(f"- results/trento_combined_results.png")
        print(f"- results/trento_combined_clusters.npy")
        print(f"- parts_trento_clusters/")
        print("="*80)
    else:
        print("No valid parts processed.")

if __name__ == "__main__":
    main()
