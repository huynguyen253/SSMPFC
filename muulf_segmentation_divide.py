# Image Segmentation with Multi-part Division
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

def divide_into_parts(image, labels=None, n_rows=3, n_cols=3):
    """Divide image into n_rows x n_cols parts"""
    height, width = image.shape[:2]
    parts = []
    label_parts = []
    
    # Calculate part dimensions
    part_height = height // n_rows
    part_width = width // n_cols
    
    # Generate part names
    part_names = []
    for i in range(n_rows):
        for j in range(n_cols):
            if i == 0:
                row_name = "top"
            elif i == n_rows - 1:
                row_name = "bottom"
            else:
                row_name = f"middle_{i}"
                
            if j == 0:
                col_name = "left"
            elif j == n_cols - 1:
                col_name = "right"
            else:
                col_name = f"center_{j}"
                
            part_names.append(f"{row_name}_{col_name}")
    
    # Indices for each part
    positions = []
    for i in range(n_rows):
        for j in range(n_cols):
            positions.append((i,j))
    
    for idx, (i, j) in enumerate(positions):
        # Calculate start and end indices for this part
        start_h = i * part_height
        end_h = (i + 1) * part_height if i < n_rows - 1 else height
        start_w = j * part_width
        end_w = (j + 1) * part_width if j < n_cols - 1 else width
        
        # Extract this part
        part = image[start_h:end_h, start_w:end_w]
        parts.append({
            'data': part,
            'position': (i, j),
            'indices': (start_h, end_h, start_w, end_w),
            'name': part_names[idx]
        })
        
        # Also divide labels if available
        if labels is not None:
            label_part = labels[start_h:end_h, start_w:end_w]
            label_parts.append({
                'data': label_part,
                'position': (i, j),
                'indices': (start_h, end_h, start_w, end_w),
                'name': part_names[idx]
            })
    
    return parts, label_parts if labels is not None else None

def prepare_data_for_processing(hsi_part, rgb_part, labels_part=None):
    """Prepare the data for processing algorithms"""
    height, width = hsi_part.shape[:2]
    
    # Reshape to 2D for processing (pixels x features)
    hsi_reshaped = hsi_part.reshape(-1, hsi_part.shape[-1])  # (height*width, bands)
    
    # Create position matrix for each pixel
    pixel_positions = np.zeros((height, width, 2), dtype=int)
    for i in range(height):
        for j in range(width):
            pixel_positions[i, j] = [i, j]
    pixel_positions_flat = pixel_positions.reshape(-1, 2)
    
    # Reshape RGB data
    if rgb_part is not None:
        # Check RGB data shape and ensure it can be properly reshaped
        print(f"RGB part shape: {rgb_part.shape}")
        # Make sure we can reshape it correctly - RGB should have shape (height, width, 3)
        if len(rgb_part.shape) == 3 and rgb_part.shape[2] == 3:
            rgb_reshaped = rgb_part.reshape(-1, 3)
        else:
            print("Warning: RGB data has unexpected shape. Using random values instead.")
            rgb_reshaped = np.random.rand(height*width, 3)  # Fallback random data
    else:
        rgb_reshaped = np.random.rand(height*width, 3)  # Fallback random data
    
    # Filter valid points if we have labels
    if labels_part is not None:
        labels_flat = labels_part.flatten()
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
        mask_2d = mask.reshape(labels_part.shape)
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

def process_part(data_views, labels, c, n_anchors, params):
    """Process one image part"""
    L = len(data_views)  # Number of views
    N = len(data_views[0])  # Number of samples
    
    print(f"\nProcessing image part with {N} samples")
    
    # Normalize multiview data
    data = normalize_multiview_data3(data_views)
    
    # Apply PCA to reduce HSI dimensions
    print("Applying PCA to reduce HSI dimensions...")
    pca = PCA(n_components=min(20, data[1].shape[1]))
    data[1] = pca.fit_transform(data[1])
    print(f"HSI shape after PCA: {data[1].shape}")
    
    # Reduce dimensionality of data to avoid memory issues
    # For smaller parts, we can use more aggressive dimensionality reduction
    print("Applying additional dimensionality reduction to avoid memory issues...")
    
    # Target dimensions - adjust based on memory constraints
    target_dim = min(15, data[0].shape[1], data[1].shape[1])
    
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
                except Exception as e:
                    print(f"Error calculating supervised metrics: {e}")
    
    return cluster_predict, cluster_predict_adj, metrics, metrics_adj

def visualize_part_results(original_shape, hsi_data, rgb_data, parts_results, n_rows=3, n_cols=3):
    """Save clustering results to files without visualization"""
    height, width = original_shape
    
    # Create empty images for the assembled results
    full_cluster_image = np.full((height, width), -1, dtype=int)
    full_cluster_image_adj = np.full((height, width), -1, dtype=int)  # For adjusted clusters u*(1-xi)
    
    # Assemble the clustering results
    for part_result in parts_results:
        cluster_predict = part_result['cluster_predict']
        cluster_predict_adj = part_result['cluster_predict_adj']  # Get adjusted clusters
        part_indices = part_result['indices']
        pixel_positions = part_result['pixel_positions']
        
        start_h, end_h, start_w, end_w = part_indices
        part_height = end_h - start_h
        part_width = end_w - start_w
        
        # Create part images for both standard and adjusted clusters
        part_image = np.full((part_height, part_width), -1, dtype=int)
        part_image_adj = np.full((part_height, part_width), -1, dtype=int)
        
        # Fill in the part images
        for i, pos in enumerate(pixel_positions):
            local_row, local_col = pos
            if 0 <= local_row < part_height and 0 <= local_col < part_width:
                part_image[local_row, local_col] = cluster_predict[i]
                part_image_adj[local_row, local_col] = cluster_predict_adj[i]
        
        # Place the parts in the full images
        full_cluster_image[start_h:end_h, start_w:end_w] = part_image
        full_cluster_image_adj[start_h:end_h, start_w:end_w] = part_image_adj
    
    # Calculate statistics for reporting
    diff_mask = full_cluster_image != full_cluster_image_adj
    percentage_diff = 100 * np.sum(diff_mask) / np.prod(diff_mask.shape)
    print(f"Difference between u and u*(1-xi) predictions: {percentage_diff:.2f}% of pixels")
    
    # Create directory for results
    os.makedirs('results', exist_ok=True)
    
    # Save basic images directly without displaying
    # Save standard clustering result - convert to float và chuẩn hóa
    full_cluster_float = full_cluster_image.astype(float)
    if full_cluster_float.max() != full_cluster_float.min():  # Tránh chia cho 0
        full_cluster_float = (full_cluster_float - full_cluster_float.min()) / (full_cluster_float.max() - full_cluster_float.min())
    plt.imsave('results/muufl_mpfcm_parts_u.png', full_cluster_float, cmap='turbo')
    
    # Save adjusted clustering result - convert to float và chuẩn hóa
    full_cluster_adj_float = full_cluster_image_adj.astype(float)
    if full_cluster_adj_float.max() != full_cluster_adj_float.min():  # Tránh chia cho 0
        full_cluster_adj_float = (full_cluster_adj_float - full_cluster_adj_float.min()) / (full_cluster_adj_float.max() - full_cluster_adj_float.min())
    plt.imsave('results/muufl_mpfcm_parts_u_xi.png', full_cluster_adj_float, cmap='turbo')
    
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
    plt.imsave('results/muufl_mpfcm_parts_difference.png', diff_image)
    
    # Save segmentation data
    np.save('results/muufl_mpfcm_labels_parts_3x3.npy', full_cluster_image)
    np.save('results/muufl_mpfcm_labels_parts_3x3_adj.npy', full_cluster_image_adj)
    
    print(f"Images saved to:")
    print(f"- results/muufl_mpfcm_parts_u.png (standard clustering)")
    print(f"- results/muufl_mpfcm_parts_u_xi.png (adjusted clustering)")
    print(f"- results/muufl_mpfcm_parts_difference.png (difference visualization)")
    
    return full_cluster_image, full_cluster_image_adj

def save_part_data(part_name, hsi_part, rgb_part, labels_part, indices):
    """Save part data to disk for further analysis"""
    # Create directory for this part
    part_dir = os.path.join("parts", part_name)
    os.makedirs(part_dir, exist_ok=True)
    
    # Save data
    np.save(os.path.join(part_dir, "hsi_data.npy"), hsi_part)
    if rgb_part is not None:
        np.save(os.path.join(part_dir, "rgb_data.npy"), rgb_part)
    if labels_part is not None:
        np.save(os.path.join(part_dir, "labels.npy"), labels_part)
    
    # Save metadata
    with open(os.path.join(part_dir, "metadata.txt"), "w") as f:
        f.write(f"Part: {part_name}\n")
        f.write(f"Indices: {indices}\n")
        f.write(f"HSI shape: {hsi_part.shape}\n")
        if rgb_part is not None:
            f.write(f"RGB shape: {rgb_part.shape}\n")
        if labels_part is not None:
            f.write(f"Labels shape: {labels_part.shape}\n")
            f.write(f"Unique labels: {np.unique(labels_part)}\n")

def load_part_data(part_name):
    """Load part data from disk"""
    part_dir = os.path.join("parts", part_name)
    
    # Check if directory exists
    if not os.path.exists(part_dir):
        return None, None, None
    
    # Load data
    hsi_data = np.load(os.path.join(part_dir, "hsi_data.npy"))
    
    # Try to load RGB data
    try:
        rgb_data = np.load(os.path.join(part_dir, "rgb_data.npy"))
        # Check if RGB data has the correct format (height, width, 3)
        if len(rgb_data.shape) == 3 and rgb_data.shape[2] == 3:
            print(f"Loaded RGB data with shape {rgb_data.shape}")
        elif len(rgb_data.shape) == 1:
            # Try to reshape based on HSI data shape
            h, w = hsi_data.shape[:2]
            if rgb_data.size == h * w * 3:
                rgb_data = rgb_data.reshape(h, w, 3)
                print(f"Reshaped RGB data to {rgb_data.shape}")
            else:
                print(f"Warning: RGB data has unexpected shape {rgb_data.shape} and size {rgb_data.size}")
                print(f"Expected size for {h}x{w} image would be {h*w*3}")
                rgb_data = None
    except FileNotFoundError:
        rgb_data = None
    except Exception as e:
        print(f"Error loading RGB data: {e}")
        rgb_data = None
    
    # Try to load labels
    try:
        labels = np.load(os.path.join(part_dir, "labels.npy"))
    except FileNotFoundError:
        labels = None
    
    return hsi_data, rgb_data, labels

def main():
    print("Starting MUUFL 3x3 Parts Image Segmentation using MPFCM with MV_DAGL...")
    
    # Load MUUFL data
    filepath = r"D:\mpfcm_8_6_12 (1)\mpfcm_8_6_12\data\muufl_gulfport_campus_1_hsi_220_label.mat"
    hsi_data, rgb_data, labels, has_labels = load_muufl_data(filepath)
    
    if hsi_data is None:
        print("Error loading data. Exiting.")
        return
    
    # Get original dimensions
    height, width = hsi_data.shape[:2]
    print(f"Original image dimensions: {height}x{width}")
    
    # Define the number of parts in each dimension
    n_rows = 3
    n_cols = 3
    
    # Divide image into parts (3x3 grid)
    print(f"Dividing image into {n_rows}x{n_cols} parts...")
    hsi_parts, label_parts = divide_into_parts(hsi_data, labels, n_rows, n_cols)
    
    if rgb_data is not None:
        print(f"RGB data shape before dividing: {rgb_data.shape}")
        if len(rgb_data.shape) == 3 and rgb_data.shape[2] == 3:
            rgb_parts, _ = divide_into_parts(rgb_data, None, n_rows, n_cols)
        else:
            print(f"RGB data has unexpected shape: {rgb_data.shape}. Will use random RGB values.")
            rgb_parts = [None] * len(hsi_parts)
    else:
        rgb_parts = [None] * len(hsi_parts)
    
    # Create output directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('parts', exist_ok=True)
    
    # Parameters
    c = 11  # Number of clusters
    n_anchors = c  # Number of anchors
    
    # Set parameters for MPFCM
    alpha = 20
    beta = alpha/(100 * 5)  # Adjusted for smaller part sizes
    sigma = 5
    omega = 1
    theta = 0.0005
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
    
    # Process each part
    parts_results = []
    
    for idx, (hsi_part, rgb_part) in enumerate(zip(hsi_parts, rgb_parts)):
        part_name = hsi_part['name']
        print(f"\n\n{'='*50}")
        print(f"Processing part {idx+1}/{len(hsi_parts)}: {part_name}")
        print(f"Position: {hsi_part['position']}")
        print(f"Indices: {hsi_part['indices']}")
        print(f"Shape: {hsi_part['data'].shape}")
        
        # Check if we can load pre-existing data for this part
        existing_hsi, existing_rgb, existing_labels = load_part_data(part_name)
        
        if existing_hsi is None:
            # Save part data for further analysis
            print(f"Saving {part_name} data to disk...")
            save_part_data(
                part_name, 
                hsi_part['data'], 
                rgb_part['data'] if rgb_part is not None else None,
                label_parts[idx]['data'] if label_parts is not None else None,
                hsi_part['indices']
            )
            
            # Use the newly created data
            current_hsi = hsi_part['data']
            current_rgb = rgb_part['data'] if rgb_part is not None else None
            current_labels = label_parts[idx]['data'] if label_parts is not None else None
        else:
            print(f"Using existing data for {part_name} from disk...")
            current_hsi = existing_hsi
            current_rgb = existing_rgb
            current_labels = existing_labels
        
        # Prepare data for processing
        data_views, part_shape, mask_2d, labels_valid, pixel_positions = prepare_data_for_processing(
            current_hsi,
            current_rgb,
            current_labels
        )
        
        # Process this part
        cluster_predict, cluster_predict_adj, metrics, metrics_adj = process_part(
            data_views, labels_valid, c, n_anchors, params
        )
        
        # Store results
        parts_results.append({
            'part_idx': idx,
            'position': hsi_part['position'],
            'indices': hsi_part['indices'],
            'name': part_name,
            'cluster_predict': cluster_predict,
            'cluster_predict_adj': cluster_predict_adj,
            'metrics': metrics,
            'metrics_adj': metrics_adj,
            'pixel_positions': pixel_positions
        })
    
    # Visualize and save the combined results
    full_cluster_image, full_cluster_image_adj = visualize_part_results(
        (height, width), hsi_data, rgb_data, parts_results, n_rows, n_cols
    )
    
    print(f"\n3x3 Parts segmentation completed!")
    print(f"Results saved to:")
    print(f"- results/muufl_mpfcm_comparison_u_xi.png (comparison visualization)")
    print(f"- results/muufl_mpfcm_labels_parts_3x3.npy (regular clustering)")
    print(f"- results/muufl_mpfcm_labels_parts_3x3_adj.npy (adjusted clustering)")
    print(f"- Individual part data saved in parts/ directory")
    
    verification = np.copy(rgb_data) if rgb_data is not None else np.zeros((height, width, 3))
    
    # Chuyển đổi sang float nếu cần và chuẩn hóa sang [0,1]
    if verification.dtype != np.float32 and verification.dtype != np.float64:
        verification = verification.astype(np.float32)
        if verification.max() > 1.0:  # Nếu giá trị > 1, giả sử khoảng [0,255]
            verification = verification / 255.0
    
    # Draw horizontal borders - sử dụng [1,0,0] cho đỏ (normalized)
    for i in range(1, n_rows):
        border_y = i * (height // n_rows)
        verification[border_y-1:border_y+1, :] = [1.0, 0.0, 0.0]  # Red horizontal lines
    
    # Draw vertical borders
    for j in range(1, n_cols):
        border_x = j * (width // n_cols)
        verification[:, border_x-1:border_x+1] = [1.0, 0.0, 0.0]  # Red vertical lines
    
    # Đảm bảo tất cả giá trị nằm trong khoảng [0,1]
    verification = np.clip(verification, 0.0, 1.0)
    
    # Save verification image directly
    os.makedirs('parts', exist_ok=True)
    plt.imsave('parts/verification.png', verification)
    print("- parts/verification.png (grid visualization)")

if __name__ == "__main__":
    main()