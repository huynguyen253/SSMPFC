# muulf_segmentation_resized_full.py
import numpy as np
import os
import time
import cv2
from scipy.linalg import svd
import matplotlib.pyplot as plt
from pymatreader import read_mat
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.decomposition import PCA
from scipy import signal
from scipy.optimize import linear_sum_assignment

# Import các module cần thiết từ project
from mpfcm import run_MPFCM, pred_cluster
from MV_DAGL import MV_DAGL
from ELMSC.elmsc import elmsc
from ELMSC.augmented_data_matrix import compute_augmented_data_matrix
from ELMSC.utils import get_Aff
from normalization import normalize_multiview_data3

def _find_nested_array_by_key_substring(container, substrings):
    """Search nested dict-like structures for first numpy-like array whose key
    contains any of substrings (case-insensitive). Returns array or None.
    """
    try:
        items = container.items() if hasattr(container, 'items') else []
    except Exception:
        items = []
    for k, v in items:
        try:
            key = str(k).lower()
        except Exception:
            key = ""
        if any(s in key for s in substrings):
            try:
                arr = np.array(v)
                return arr
            except Exception:
                pass
        # Recurse
        if hasattr(v, 'items'):
            found = _find_nested_array_by_key_substring(v, substrings)
            if found is not None:
                return found
    return None

def resize_multichannel_image(image, new_size, interpolation=cv2.INTER_CUBIC):
    """Resize a multichannel image by applying resize to each channel separately"""
    image = image * 255
    if len(image.shape) == 3:
        return np.stack([
            cv2.resize(image[..., i], new_size, interpolation=interpolation)
            for i in range(image.shape[-1])
        ], axis=-1)
    else:
        return cv2.resize(image, new_size, interpolation=interpolation)/255

def load_muufl_data_resized(filepath, new_size=(164, 110)):
    """Load MUUFL data from .mat file, resize and prepare for processing"""
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
    
    # Resize HSI data
    print(f"Resizing HSI data to {new_size}...")
    resized_hsi = resize_multichannel_image(hsi_filtered, new_size)
    print(f"Resized HSI shape: {resized_hsi.shape}")
    
    # Resize RGB data if available
    if rgb is not None:
        print(f"Resizing RGB data to {new_size}...")
        # Save original RGB before resizing
        original_rgb = np.copy(rgb)
        resized_rgb = resize_multichannel_image(rgb, new_size)
        print(f"Resized RGB shape: {resized_rgb.shape}")
        
        # Display comparison between original and resized RGB
        plt.figure(figsize=(12, 6))
        
        # Print min/max values for debugging
        print(f"Original RGB min: {original_rgb.min()}, max: {original_rgb.max()}")
        print(f"Resized RGB min: {resized_rgb.min()}, max: {resized_rgb.max()}")
        
        # Original RGB - improved normalization
        plt.subplot(1, 2, 1)
        # Clip values to valid range
        original_rgb_clipped = np.clip(original_rgb, 0, 255) if original_rgb.max() > 1 else np.clip(original_rgb * 255, 0, 255)
        # Ensure correct normalization for visualization
        original_rgb_vis = original_rgb_clipped / 255.0
        plt.imshow(original_rgb_vis)
        plt.title(f'Original RGB Image: {original_rgb.shape[:2]}')
        plt.axis('off')
        
        # Resized RGB - improved normalization
        plt.subplot(1, 2, 2)
        # Clip values to valid range
        resized_rgb_clipped = np.clip(resized_rgb, 0, 255) if resized_rgb.max() > 1 else np.clip(resized_rgb * 255, 0, 255)
        # Ensure correct normalization for visualization
        resized_rgb_vis = resized_rgb_clipped / 255.0
        plt.imshow(resized_rgb_vis)
        plt.title(f'Resized RGB Image: {resized_rgb.shape[:2]}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('results/rgb_resize_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        resized_rgb = None
    
    # Extract and resize labels if available
    try:
        labels = hsi["sceneLabels"]["labels"]
        print(f"Original labels shape: {labels.shape}")
        
        # Resize labels using nearest neighbor interpolation
        resized_labels = cv2.resize(labels, new_size, interpolation=cv2.INTER_NEAREST)
        resized_labels = np.round(resized_labels).astype(np.int32)
        print(f"Resized labels shape: {resized_labels.shape}")
        has_labels = True
    except (KeyError, IndexError):
        print("No label information found in the data")
        has_labels = False
        resized_labels = None
    
    # Try to extract LiDAR/DSM (multi/single-channel) and resize
    try:
        lidar_raw = None
        if isinstance(hsi, dict) and 'Lidar' in hsi:
            lidar_field = hsi['Lidar']
            # pymatreader converts MATLAB cell to Python list
            if isinstance(lidar_field, (list, tuple)):
                lidar_bands = []
                for elem in lidar_field:
                    arr = np.array(elem)
                    if arr.ndim == 3 and arr.shape[-1] == 1:
                        arr = arr[..., 0]
                    if arr.ndim == 2:
                        lidar_bands.append(arr)
                if len(lidar_bands) > 0:
                    lidar_raw = np.stack(lidar_bands, axis=-1)  # (H,W,C_lidar)
            else:
                arr = np.array(lidar_field)
                if arr.ndim == 2:
                    lidar_raw = arr[..., None]
                elif arr.ndim == 3:
                    lidar_raw = arr
        # Fallback: search by key names
        if lidar_raw is None:
            lidar_candidates = ['lidar', 'dsm', 'elevation', 'height']
            lidar_raw = _find_nested_array_by_key_substring(hsi, lidar_candidates)
            if lidar_raw is None:
                lidar_raw = _find_nested_array_by_key_substring(data, lidar_candidates)
            if lidar_raw is not None and lidar_raw.ndim == 2:
                lidar_raw = lidar_raw[..., None]
        if lidar_raw is not None:
            # Normalize lidar_raw into numpy array with explicit channel dim
            if isinstance(lidar_raw, (list, tuple)):
                lidar_bands = []
                for elem in lidar_raw:
                    arr = np.array(elem)
                    if arr.ndim == 3 and arr.shape[-1] == 1:
                        arr = arr[..., 0]
                    if arr.ndim == 2:
                        lidar_bands.append(arr)
                if len(lidar_bands) == 0:
                    raise ValueError("Empty/unsupported LiDAR cell contents")
                lidar_raw = np.stack(lidar_bands, axis=-1)
            else:
                lidar_raw = np.array(lidar_raw)
                if lidar_raw.ndim == 2:
                    lidar_raw = lidar_raw[..., None]
                elif lidar_raw.ndim != 3:
                    raise ValueError(f"Unsupported LiDAR ndim: {lidar_raw.ndim}")

            print(f"Found LiDAR with shape: {lidar_raw.shape}")
            resized_lidar = resize_multichannel_image(lidar_raw, new_size)
            print(f"Resized LiDAR shape: {resized_lidar.shape}")
        else:
            print("LiDAR not found; skipping LiDAR view.")
            resized_lidar = None
    except Exception as e:
        print(f"LiDAR extraction error: {e}; skipping LiDAR view.")
        resized_lidar = None

    # Reshape to 2D for processing (pixels x features)
    height, width = resized_hsi.shape[:2]
    hsi_reshaped = resized_hsi.reshape(-1, resized_hsi.shape[-1])  # (height*width, bands)
    
    # Create position matrix for each pixel
    pixel_positions = np.zeros((height, width, 2), dtype=int)
    for i in range(height):
        for j in range(width):
            pixel_positions[i, j] = [i, j]
    pixel_positions_flat = pixel_positions.reshape(-1, 2)
    
    # Reshape RGB data
    if resized_rgb is not None:
        rgb_reshaped = resized_rgb.reshape(-1, 3)
    else:
        rgb_reshaped = np.random.rand(height*width, 3)  # Fallback random data
    
    # Reshape LiDAR data (single channel) if available
    if resized_lidar is not None:
        if resized_lidar.ndim == 2:
            lidar_reshaped = resized_lidar.reshape(-1, 1)
        else:
            lidar_reshaped = resized_lidar.reshape(-1, resized_lidar.shape[-1])
    else:
        lidar_reshaped = None

    # Filter valid points if we have labels
    if has_labels:
        labels_flat = resized_labels.flatten()
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
        mask_2d = mask.reshape(resized_labels.shape)
    else:
        mask_2d = np.ones((height, width), dtype=bool)
        labels_valid = None
        pixel_positions_valid = pixel_positions_flat
    
    # Normalize data
    scaler_rgb = StandardScaler()
    scaler_hsi = StandardScaler()
    scaler_lidar = StandardScaler() if lidar_reshaped is not None else None
    
    rgb_normalized = scaler_rgb.fit_transform(rgb_reshaped)
    hsi_normalized = scaler_hsi.fit_transform(hsi_reshaped)
    lidar_normalized = scaler_lidar.fit_transform(lidar_reshaped) if lidar_reshaped is not None else None
    
    # Format data for algorithms
    data_views = [rgb_normalized, hsi_normalized]
    if lidar_normalized is not None:
        data_views.append(lidar_normalized)
    
    return data_views, (height, width), resized_rgb, resized_hsi, resized_lidar, mask_2d, labels_valid, pixel_positions_valid

def process_full_image(data_views, labels, c, n_anchors, params):
    """Process the entire image"""
    L = len(data_views)  # Number of views
    N = len(data_views[0])  # Number of samples
    
    print(f"\nProcessing image with {N} samples")
    
    # Normalize multiview data
    data = normalize_multiview_data3(data_views)
    print(data)
    # Apply PCA to reduce HSI dimensions
    print("Applying PCA to reduce HSI dimensions...")
    pca = PCA(n_components=min(20, data[1].shape[1]))
    data[1] = pca.fit_transform(data[1])
    print(f"HSI shape after PCA: {data[1].shape}")
    
    # Calculate augmented matrix for ELMSC (optional) — skip if not exactly 2 views
    try:
        if len(data) == 2:
            print("Calculating augmented matrix (2 views)...")
            augmented_matrix = compute_augmented_data_matrix(data)
            print(f"Augmented matrix shape: {augmented_matrix.shape}")
        else:
            print("Skipping augmented matrix (ELMSC) because number of views != 2")
    except Exception as e:
        print(f"Skipping augmented matrix due to error: {e}")
    
    # Run MV_DAGL
    print("Running MV_DAGL...")
    start_time = time.time()
    z_c, z_list = MV_DAGL(data, n_anchors, max_iter=1000, alpha=0.01, tol=1e-6)
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
    # Check if we have enough data for evaluation
    if len(cluster_predict_adj) > 1:
        # Check if we have more than one unique cluster
        unique_clusters_adj = np.unique(cluster_predict_adj)
        if len(unique_clusters_adj) > 1:
            try:
                # Unsupervised metrics
                silhouette_adj = silhouette_score(combined_features, cluster_predict_adj)
                metrics_adj["silhouette"] = silhouette_adj
                
                calinski_adj = calinski_harabasz_score(combined_features, cluster_predict_adj)
                metrics_adj["calinski_harabasz"] = calinski_adj
                
                davies_adj = davies_bouldin_score(combined_features, cluster_predict_adj)
                metrics_adj["davies_bouldin"] = davies_adj
                
                print(f"Clustering Quality:")
                print(f"  - Silhouette Score: {silhouette_adj:.4f} (higher is better)")
                print(f"  - Calinski-Harabasz Index: {calinski_adj:.4f} (higher is better)")
                print(f"  - Davies-Bouldin Index: {davies_adj:.4f} (lower is better)")
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
            print("\nSupervised metrics for standard clustering (from u):")
            print(f"  - Adjusted Rand Index: {ari:.4f} (higher is better)")
            print(f"  - Normalized Mutual Information: {nmi:.4f} (higher is better)")

            # Confusion matrix and per-class F1 for standard clustering (with label mapping)
            # Map predicted cluster ids to ground-truth class ids using Hungarian + fallback
            true_labels_unique = np.unique(valid_labels)
            pred_labels_unique = np.unique(valid_clusters)
            # Build contingency for mapping
            contingency = np.zeros((len(true_labels_unique), len(pred_labels_unique)), dtype=np.int64)
            true_index = {lbl: i for i, lbl in enumerate(true_labels_unique)}
            pred_index = {lbl: j for j, lbl in enumerate(pred_labels_unique)}
            for t, p in zip(valid_labels, valid_clusters):
                contingency[true_index[t], pred_index[p]] += 1
            # Hungarian for primary matches
            row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)
            mapping = {pred_labels_unique[j]: true_labels_unique[i] for i, j in zip(row_ind, col_ind)}
            # Fallback mapping for any remaining predicted clusters
            unmatched_pred = set(range(len(pred_labels_unique))) - set(col_ind)
            for j in unmatched_pred:
                best_i = int(np.argmax(contingency[:, j]))
                mapping[pred_labels_unique[j]] = true_labels_unique[best_i]
            # Apply mapping
            mapped_valid_clusters = np.array([mapping[p] for p in valid_clusters])
            # Compute confusion matrix and per-class F1
            ordered_labels = list(np.sort(true_labels_unique))
            cm_std = confusion_matrix(valid_labels, mapped_valid_clusters, labels=ordered_labels)
            f1_std = f1_score(valid_labels, mapped_valid_clusters, labels=ordered_labels, average=None, zero_division=0)
            # Print brief summary
            print("  - Confusion matrix (standard) computed. Shape:", cm_std.shape)
            print("  - F1 per class (standard):")
            for lbl, f1v in zip(ordered_labels, f1_std):
                print(f"    class {lbl}: {f1v:.4f}")
            # Save to results
            import json  # local import to avoid top-level changes
            os.makedirs('results', exist_ok=True)
            with open('results/confusion_matrix_standard.json', 'w') as f:
                json.dump({"labels": [int(x) for x in ordered_labels], "matrix": cm_std.tolist()}, f, indent=2)
            with open('results/per_class_f1_standard.json', 'w') as f:
                json.dump({str(int(lbl)): float(val) for lbl, val in zip(ordered_labels, f1_std)}, f, indent=2)
            
            # Adjusted clustering metrics
            valid_clusters_adj = cluster_predict_adj[valid_indices]
            ari_adj = adjusted_rand_score(valid_labels, valid_clusters_adj)
            metrics_adj["ari"] = ari_adj
            nmi_adj = normalized_mutual_info_score(valid_labels, valid_clusters_adj)
            metrics_adj["nmi"] = nmi_adj
            print("\nSupervised metrics for adjusted clustering (from u*(1-xi)):")
            print(f"  - Adjusted Rand Index: {ari_adj:.4f} (higher is better)")
            print(f"  - Normalized Mutual Information: {nmi_adj:.4f} (higher is better)")

            # Confusion matrix and per-class F1 for adjusted clustering (with label mapping)
            pred_labels_unique_adj = np.unique(valid_clusters_adj)
            contingency_adj = np.zeros((len(true_labels_unique), len(pred_labels_unique_adj)), dtype=np.int64)
            pred_index_adj = {lbl: j for j, lbl in enumerate(pred_labels_unique_adj)}
            for t, p in zip(valid_labels, valid_clusters_adj):
                contingency_adj[true_index[t], pred_index_adj[p]] += 1
            row_ind_adj, col_ind_adj = linear_sum_assignment(contingency_adj.max() - contingency_adj)
            mapping_adj = {pred_labels_unique_adj[j]: true_labels_unique[i] for i, j in zip(row_ind_adj, col_ind_adj)}
            unmatched_pred_adj = set(range(len(pred_labels_unique_adj))) - set(col_ind_adj)
            for j in unmatched_pred_adj:
                best_i = int(np.argmax(contingency_adj[:, j]))
                mapping_adj[pred_labels_unique_adj[j]] = true_labels_unique[best_i]
            mapped_valid_clusters_adj = np.array([mapping_adj[p] for p in valid_clusters_adj])
            cm_adj = confusion_matrix(valid_labels, mapped_valid_clusters_adj, labels=ordered_labels)
            f1_adj = f1_score(valid_labels, mapped_valid_clusters_adj, labels=ordered_labels, average=None, zero_division=0)
            print("  - Confusion matrix (adjusted) computed. Shape:", cm_adj.shape)
            print("  - F1 per class (adjusted):")
            for lbl, f1v in zip(ordered_labels, f1_adj):
                print(f"    class {lbl}: {f1v:.4f}")
            with open('results/confusion_matrix_adjusted.json', 'w') as f:
                json.dump({"labels": [int(x) for x in ordered_labels], "matrix": cm_adj.tolist()}, f, indent=2)
            with open('results/per_class_f1_adjusted.json', 'w') as f:
                json.dump({str(int(lbl)): float(val) for lbl, val in zip(ordered_labels, f1_adj)}, f, indent=2)
            
            # Compare the two
            print("\nComparison:")
            print(f"  - ARI Improvement: {ari_adj-ari:.4f} ({(ari_adj-ari)/ari*100:.2f}%)")
            print(f"  - NMI Improvement: {nmi_adj-nmi:.4f} ({(nmi_adj-nmi)/nmi*100:.2f}%)")
        except Exception as e:
            print(f"Error calculating supervised metrics: {e}")
    
    return cluster_predict, cluster_predict_adj, metrics, metrics_adj

def main():
    print("Starting Resized MUUFL Full-Image Segmentation using MPFCM with MV_DAGL and ELMSC...")
    
    # Set target size for resized data - 1/4 of original in each dimension
    # Original size: (327, 220) [height, width]
    # Reduced to 1/4: (82, 55) [height, width]
    # OpenCV expects (width, height) order
    new_size = (55, 82)  # width, height for OpenCV functions
    new_size2 = (110,164)
    new_size3 = (165, 246)
    # Load and resize MUUFL data
    #filepath = "/kaggle/input/mpfcm/pytorch/default/2/data/muufl_gulfport_campus_1_hsi_220_label.mat"

    filepath = r"D:\mpfcm_8_6_12 (1)\mpfcm_8_6_12\data\muufl_gulfport_campus_1_hsi_220_label.mat"
    data_views, original_shape, resized_rgb, resized_hsi, resized_lidar, mask_2d, labels, pixel_positions = load_muufl_data_resized(filepath, new_size2)
    
    if data_views is None:
        print("Error loading data. Exiting.")
        return
    
    # Parameters
    c = 11  # Number of clusters
    L = len(data_views)  # Number of views (now includes LiDAR)
    N = len(data_views[0])  # Number of samples
    n_anchors = c  # Number of anchors
    
    print(f"\nParameters:")
    print(f"Number of views: {L}")
    print(f"Number of clusters: {c}")
    print(f"Number of samples: {N} (after filtering)")
    print(f"Number of anchors: {n_anchors}")
    
    # Set parameters for MPFCM
    alpha = 20
    beta = alpha/(100 * 15 *6)
    sigma = 5
    omega = 0.001
    sump = 1 - alpha - beta - sigma - omega
    theta = 0.0003
    rho = 1

    params = {
        "alpha": alpha,
        "beta": beta,
        "sigma": sigma,
        "theta": theta,
        "omega": omega,
        "rho": rho
    }
    
    # Create an empty image for the final result
    height, width = resized_hsi.shape[:2]  # Use the actual dimensions from loaded data
    final_cluster_image = np.full((height, width), -1, dtype=int)
    
    # Process the entire image
    print("\nProcessing the entire image...")
    cluster_predict, cluster_predict_adj, metrics, metrics_adj = process_full_image(data_views, labels, c, n_anchors, params)
    
    print(f"Final cluster image shape: {final_cluster_image.shape}")
    print(f"Number of pixel positions: {len(pixel_positions)}")
    print(f"Number of clusters predicted: {len(cluster_predict)}")
    
    # Place clusters back into the final image
    for i, (row, col) in enumerate(pixel_positions):
        if row < height and col < width:  # Add safety check
            final_cluster_image[row, col] = cluster_predict[i]
        else:
            print(f"Warning: Position ({row}, {col}) out of bounds for image size {height}x{width}")
    
    # Create two final cluster images - one for standard clustering and one for adjusted
    final_cluster_image_std = np.full((height, width), -1, dtype=int)  # Use the height/width from above
    final_cluster_image_adj = np.full((height, width), -1, dtype=int)  # Use the height/width from above
    
    print(f"Creating two final cluster images...")
    print(f"Number of pixel positions: {len(pixel_positions)}")
    print(f"Standard clusters (from u): {len(cluster_predict)} points")
    print(f"Adjusted clusters (from u*(1-xi)): {len(cluster_predict_adj)} points")
    
    # Place standard clusters back into the first image
    for i, (row, col) in enumerate(pixel_positions):
        if row < height and col < width:  # Add safety check
            final_cluster_image_std[row, col] = cluster_predict[i]
        else:
            print(f"Warning: Position ({row}, {col}) out of bounds for image size {height}x{width}")
    
    # Place adjusted clusters back into the second image
    for i, (row, col) in enumerate(pixel_positions):
        if row < height and col < width:  # Add safety check
            final_cluster_image_adj[row, col] = cluster_predict_adj[i]
        else:
            print(f"Warning: Position ({row}, {col}) out of bounds for image size {height}x{width}")
    
    # Compute binary disagreement heatmap against ground truth (using adjusted clustering)
    # Map adjusted predicted clusters to ground-truth labels via Hungarian assignment
    try:
        true_labels_unique = np.unique(labels)
        pred_labels_unique_adj = np.unique(cluster_predict_adj)
        contingency_adj = np.zeros((len(true_labels_unique), len(pred_labels_unique_adj)), dtype=np.int64)
        true_index = {lbl: i for i, lbl in enumerate(true_labels_unique)}
        pred_index_adj = {lbl: j for j, lbl in enumerate(pred_labels_unique_adj)}
        for t, p in zip(labels, cluster_predict_adj):
            contingency_adj[true_index[t], pred_index_adj[p]] += 1
        row_ind_adj, col_ind_adj = linear_sum_assignment(contingency_adj.max() - contingency_adj)
        mapping_adj = {pred_labels_unique_adj[j]: true_labels_unique[i] for i, j in zip(row_ind_adj, col_ind_adj)}
        unmatched_pred_adj = set(range(len(pred_labels_unique_adj))) - set(col_ind_adj)
        for j in unmatched_pred_adj:
            best_i = int(np.argmax(contingency_adj[:, j]))
            mapping_adj[pred_labels_unique_adj[j]] = true_labels_unique[best_i]
        mapped_clusters_adj_1d = np.array([mapping_adj[p] for p in cluster_predict_adj])
        # Build 2D ground-truth label image
        gt_image = np.full((height, width), -1, dtype=int)
        for i, (row, col) in enumerate(pixel_positions):
            if row < height and col < width:
                gt_image[row, col] = labels[i]
        # Build 2D mapped adjusted prediction image
        mapped_adj_image = np.full((height, width), -1, dtype=int)
        for i, (row, col) in enumerate(pixel_positions):
            if row < height and col < width:
                mapped_adj_image[row, col] = mapped_clusters_adj_1d[i]
        # Disagreement where both GT and prediction are available
        valid_pixel_mask = (gt_image != -1) & (mapped_adj_image != -1)
        disagreement_mask_truth = ((mapped_adj_image != gt_image) & valid_pixel_mask).astype(np.uint8)
        valid_pixels = int(np.sum(valid_pixel_mask))
        disagreement_pixels = int(np.sum(disagreement_mask_truth))
        disagreement_ratio = (disagreement_pixels / valid_pixels) if valid_pixels > 0 else 0.0
        print(f"GT disagreement pixels (adjusted vs GT): {disagreement_pixels}/{valid_pixels} ({disagreement_ratio*100:.2f}%)")
    except Exception as e:
        print(f"Error computing GT disagreement heatmap: {e}")
        disagreement_mask_truth = np.zeros((height, width), dtype=np.uint8)
    
    # Create output directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save metrics to file
    import json
    with open('results/full_image_metrics_resized.json', 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        cleaned_metrics = {}
        for metric_name, value in metrics.items():
            if isinstance(value, np.float64) or isinstance(value, np.float32):
                cleaned_metrics[metric_name] = float(value)
            else:
                cleaned_metrics[metric_name] = value
        
        json.dump(cleaned_metrics, f, indent=2)
    
    # Save adjusted metrics to file
    with open('results/full_image_metrics_resized_adj.json', 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        cleaned_metrics_adj = {}
        for metric_name, value in metrics_adj.items():
            if isinstance(value, np.float64) or isinstance(value, np.float32):
                cleaned_metrics_adj[metric_name] = float(value)
            else:
                cleaned_metrics_adj[metric_name] = value
        
        json.dump(cleaned_metrics_adj, f, indent=2)
    
    # Generate visualizations
    plt.figure(figsize=(15, 10))
    
    # Original RGB image
    plt.subplot(2, 2, 1)
    if resized_rgb is not None:
        rgb_vis = resized_rgb / 255.0 if resized_rgb.max() > 1 else resized_rgb
        plt.imshow(rgb_vis)
    else:
        plt.text(0.5, 0.5, 'RGB not available', horizontalalignment='center')
    plt.title('Resized MUUFL RGB Image')
    plt.axis('off')
    
    # Segmentation result
    plt.subplot(2, 2, 2)
    plt.imshow(final_cluster_image, cmap='turbo')
    plt.colorbar(label='Cluster')
    plt.title(f'MUUFL Segmentation using MPFCM (c={c})')
    plt.axis('off')
    
    # Generate separate visualizations for standard and adjusted clustering
    plt.figure(figsize=(15, 10))
    
    # Original RGB image
    plt.subplot(2, 3, 1)
    if resized_rgb is not None:
        rgb_vis = resized_rgb / 255.0 if resized_rgb.max() > 1 else resized_rgb
        plt.imshow(rgb_vis)
    else:
        plt.text(0.5, 0.5, 'RGB not available', horizontalalignment='center')
    plt.title('Resized MUUFL RGB Image')
    plt.axis('off')
    
    # Standard clustering result (from u)
    plt.subplot(2, 3, 2)
    plt.imshow(final_cluster_image_std, cmap='turbo')
    plt.colorbar(label='Cluster')
    plt.title(f'Standard Clustering (from u)')
    plt.axis('off')
    
    # Adjusted clustering result (from u*(1-xi))
    plt.subplot(2, 3, 3)
    plt.imshow(final_cluster_image_adj, cmap='turbo')
    plt.colorbar(label='Cluster')
    plt.title(f'Adjusted Clustering (from u*(1-xi))')
    plt.axis('off')
    
    # HSI visualization (first 3 bands as RGB)
    plt.subplot(2, 3, 4)
    hsi_rgb = resized_hsi[:, :, [0, resized_hsi.shape[2]//2, -1]]  # First, middle, last band
    hsi_rgb_normalized = (hsi_rgb - hsi_rgb.min()) / (hsi_rgb.max() - hsi_rgb.min())
    plt.imshow(hsi_rgb_normalized)
    plt.title('HSI Visualization (3 bands)')
    plt.axis('off')
    
    # Add metrics visualization for standard clustering
    plt.subplot(2, 3, 5)
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    y_pos = np.arange(len(metric_names))
    
    plt.barh(y_pos, metric_values, color='teal')
    plt.yticks(y_pos, metric_names)
    plt.xlabel('Value')
    plt.title('Standard Clustering Metrics')
    
    # Add metrics visualization for adjusted clustering
    plt.subplot(2, 3, 6)
    metric_names_adj = list(metrics_adj.keys())
    metric_values_adj = list(metrics_adj.values())
    
    y_pos = np.arange(len(metric_names_adj))
    
    plt.barh(y_pos, metric_values_adj, color='orange')
    plt.yticks(y_pos, metric_names_adj)
    plt.xlabel('Value')
    plt.title('Adjusted Clustering Metrics')
    
    plt.tight_layout()
    plt.savefig('results/muufl_mpfcm_full_image_resized_comparison.png', dpi=300, bbox_inches='tight')
    
    # Visualize and save binary disagreement heatmap (adjusted vs ground truth)
    plt.figure(figsize=(8, 6))
    plt.imshow(disagreement_mask_truth, cmap='gray', vmin=0, vmax=1)
    plt.colorbar(label='Disagreement (1=diff, 0=same)')
    plt.title('Binary Disagreement Heatmap (u*(1-xi) vs Ground Truth)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/muufl_mpfcm_disagreement_heatmap_truth_binary.png', dpi=300, bbox_inches='tight')
    
    # Also save the original visualization
    plt.figure(figsize=(15, 10))
    
    # Original RGB image
    plt.subplot(2, 2, 1)
    if resized_rgb is not None:
        rgb_vis = resized_rgb / 255.0 if resized_rgb.max() > 1 else resized_rgb
        plt.imshow(rgb_vis)
    else:
        plt.text(0.5, 0.5, 'RGB not available', horizontalalignment='center')
    plt.title('Resized MUUFL RGB Image')
    plt.axis('off')
    
    # Standard clustering result (from u)
    plt.subplot(2, 2, 2)
    plt.imshow(final_cluster_image_std, cmap='turbo')
    plt.colorbar(label='Cluster')
    plt.title(f'MUUFL Segmentation using MPFCM (c={c})')
    plt.axis('off')
    
    # HSI visualization (first 3 bands as RGB)
    plt.subplot(2, 2, 3)
    plt.imshow(hsi_rgb_normalized)
    plt.title('HSI Visualization (3 bands)')
    plt.axis('off')
    
    # Add metrics visualization
    plt.subplot(2, 2, 4)
    plt.barh(y_pos, metric_values, color='teal')
    plt.yticks(y_pos, metric_names)
    plt.xlabel('Value')
    plt.title('Clustering Metrics')
    
    plt.tight_layout()
    plt.savefig('results/muufl_mpfcm_full_image_resized.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Save adjusted-only segmentation image
    plt.figure(figsize=(8, 6))
    plt.imshow(final_cluster_image_adj, cmap='turbo')
    plt.colorbar(label='Cluster')
    plt.title('Adjusted Clustering (from u*(1-xi))')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/muufl_mpfcm_full_image_resized_adj.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save segmentation result
    np.save('results/muufl_mpfcm_labels_full_image_resized.npy', final_cluster_image_std)
    np.save('results/muufl_mpfcm_labels_full_image_resized_adj.npy', final_cluster_image_adj)
    np.save('results/muufl_mpfcm_disagreement_mask_truth_binary.npy', disagreement_mask_truth)
    
    print(f"\nSegmentation completed!")
    print(f"Results saved to:")
    print(f"- results/muufl_mpfcm_full_image_resized.png")
    print(f"- results/muufl_mpfcm_full_image_resized_comparison.png")
    print(f"- results/muufl_mpfcm_disagreement_heatmap_truth_binary.png")
    print(f"- results/muufl_mpfcm_labels_full_image_resized.npy")
    print(f"- results/muufl_mpfcm_labels_full_image_resized_adj.npy")
    print(f"- results/muufl_mpfcm_disagreement_mask_truth_binary.npy")
    print(f"- results/full_image_metrics_resized.json")
    print(f"- results/full_image_metrics_resized_adj.json")
    print(f"Resized image dimensions: {height}x{width}")
    print(f"Number of clusters: {c}")
    
    # Print summary of metrics
    print("\nStandard clustering metrics:")
    for metric_name, value in metrics.items():
        print(f"  - {metric_name}: {value:.4f}")
    
    print("\nAdjusted clustering metrics:")
    for metric_name, value in metrics_adj.items():
        print(f"  - {metric_name}: {value:.4f}")
    
    # Print comparison if both have the same metrics
    common_metrics = set(metrics.keys()) & set(metrics_adj.keys())
    if common_metrics:
        print("\nMetrics improvement (adjusted vs standard):")
        for metric in common_metrics:
            diff = metrics_adj[metric] - metrics[metric]
            pct = (diff / abs(metrics[metric])) * 100 if metrics[metric] != 0 else 0
            better = "better" if (metric != "davies_bouldin" and diff > 0) or (metric == "davies_bouldin" and diff < 0) else "worse"
            print(f"  - {metric}: {diff:.4f} ({pct:.2f}%) {better}")

if __name__ == "__main__":
    main()