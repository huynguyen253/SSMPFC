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
from scipy.optimize import linear_sum_assignment

from mpfcm import run_MPFCM, pred_cluster
from MV_DAGL import MV_DAGL
from normalization import normalize_multiview_data3

def _find_nested_array_by_key_substring(container, substrings):
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
        if hasattr(v, 'items'):
            found = _find_nested_array_by_key_substring(v, substrings)
            if found is not None:
                return found
    return None

def load_muufl_data_full(filepath):
    print(f"\nLoading MUUFL data from {filepath} (full resolution)...")
    data = read_mat(filepath)
    hsi = data["hsi"]
    rgb = hsi.get("RGB", None)

    # HSI data full-size
    hsi_data = hsi["Data"]
    print(f"HSI data shape: {hsi_data.shape}")

    # Optional band filtering (same as resized version)
    hsi_filtered = hsi_data[:, :, 4:-4] if hsi_data.shape[2] > 8 else hsi_data
    print(f"HSI filtered shape: {hsi_filtered.shape}")

    resized_hsi = hsi_filtered  # keep full resolution
    resized_rgb = rgb  # keep full resolution if present

    # Labels
    try:
        labels = hsi["sceneLabels"]["labels"]
        print(f"Labels shape: {labels.shape}")
        resized_labels = labels.astype(np.int32)
        has_labels = True
    except Exception:
        print("No label information found in the data")
        resized_labels = None
        has_labels = False

    # LiDAR/DSM extraction if available
    try:
        lidar_raw = None
        if isinstance(hsi, dict) and 'Lidar' in hsi:
            lidar_field = hsi['Lidar']
            if isinstance(lidar_field, (list, tuple)):
                lidar_bands = []
                for elem in lidar_field:
                    arr = np.array(elem)
                    if arr.ndim == 3 and arr.shape[-1] == 1:
                        arr = arr[..., 0]
                    if arr.ndim == 2:
                        lidar_bands.append(arr)
                if len(lidar_bands) > 0:
                    lidar_raw = np.stack(lidar_bands, axis=-1)
            else:
                arr = np.array(lidar_field)
                if arr.ndim == 2:
                    lidar_raw = arr[..., None]
                elif arr.ndim == 3:
                    lidar_raw = arr
        if lidar_raw is None:
            lidar_candidates = ['lidar', 'dsm', 'elevation', 'height']
            lidar_raw = _find_nested_array_by_key_substring(hsi, lidar_candidates)
            if lidar_raw is None:
                lidar_raw = _find_nested_array_by_key_substring(data, lidar_candidates)
            if lidar_raw is not None and lidar_raw.ndim == 2:
                lidar_raw = lidar_raw[..., None]
        if lidar_raw is not None:
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
            resized_lidar = lidar_raw
        else:
            print("LiDAR not found; skipping LiDAR view.")
            resized_lidar = None
    except Exception as e:
        print(f"LiDAR extraction error: {e}; skipping LiDAR view.")
        resized_lidar = None

    # Flatten to 2D
    height, width = resized_hsi.shape[:2]
    hsi_reshaped = resized_hsi.reshape(-1, resized_hsi.shape[-1])

    pixel_positions = np.zeros((height, width, 2), dtype=int)
    for i in range(height):
        for j in range(width):
            pixel_positions[i, j] = [i, j]
    pixel_positions_flat = pixel_positions.reshape(-1, 2)

    if resized_rgb is not None:
        rgb_reshaped = resized_rgb.reshape(-1, 3)
    else:
        rgb_reshaped = np.random.rand(height*width, 3)

    if resized_lidar is not None:
        if resized_lidar.ndim == 2:
            lidar_reshaped = resized_lidar.reshape(-1, 1)
        else:
            lidar_reshaped = resized_lidar.reshape(-1, resized_lidar.shape[-1])
    else:
        lidar_reshaped = None

    # Mask valid labels
    if has_labels:
        labels_flat = resized_labels.flatten()
        valid_indices = labels_flat != -1
        total_points = len(labels_flat)
        valid_points = int(np.sum(valid_indices))
        print(f"Valid labeled points: {valid_points}/{total_points}")

        rgb_reshaped = rgb_reshaped[valid_indices]
        hsi_reshaped = hsi_reshaped[valid_indices]
        labels_valid = labels_flat[valid_indices]
        pixel_positions_valid = pixel_positions_flat[valid_indices]

        mask = np.zeros(labels_flat.shape, dtype=bool)
        mask[valid_indices] = True
        mask_2d = mask.reshape(resized_labels.shape)
    else:
        mask_2d = np.ones((height, width), dtype=bool)
        labels_valid = None
        pixel_positions_valid = pixel_positions_flat

    # Normalize to float32
    scaler_rgb = StandardScaler()
    scaler_hsi = StandardScaler()
    scaler_lidar = StandardScaler() if lidar_reshaped is not None else None

    rgb_normalized = scaler_rgb.fit_transform(rgb_reshaped).astype(np.float32)
    hsi_normalized = scaler_hsi.fit_transform(hsi_reshaped).astype(np.float32)
    lidar_normalized = scaler_lidar.fit_transform(lidar_reshaped).astype(np.float32) if lidar_reshaped is not None else None

    data_views = [rgb_normalized, hsi_normalized]
    if lidar_normalized is not None:
        data_views.append(lidar_normalized)

    return data_views, (height, width), resized_rgb, resized_hsi, resized_lidar, mask_2d, labels_valid, pixel_positions_valid

def process_full_image(data_views, labels, c, n_anchors, params):
    L = len(data_views)
    N = len(data_views[0])

    print(f"\nProcessing FULL image with {N} samples")

    data = normalize_multiview_data3(data_views)
    data = [np.asarray(view, dtype=np.float32) for view in data]

    print("Applying PCA to reduce HSI dimensions...")
    pca = PCA(n_components=min(20, data[1].shape[1]))
    data[1] = pca.fit_transform(data[1]).astype(np.float32)
    print(f"HSI shape after PCA: {data[1].shape}")

    print("Running MV_DAGL...")
    start_time = time.time()
    z_c, z_list = MV_DAGL(data, n_anchors, max_iter=1000, alpha=0.01, tol=1e-6)
    end_time = time.time()
    print(f"MV_DAGL completed in {end_time - start_time:.2f} seconds")

    print("Calculating similarity matrices (may be large in RAM)...")
    data_new = []
    similarity = []

    for l in range(L):
        Z = np.asarray(z_list[l], dtype=np.float32)
        tg = (Z.T @ Z).astype(np.float32)
        similarity.append(tg)
        U, _, _ = svd(Z.T, full_matrices=False)
        data_new.append(np.asarray(U[:, :n_anchors], dtype=np.float32))

    Zc = np.asarray(z_c, dtype=np.float32)
    U, _, _ = svd(Zc.T, full_matrices=False)
    data_new.append(np.asarray(U[:, :n_anchors], dtype=np.float32))

    tg = (Zc.T @ Zc).astype(np.float32)
    similarity.append(tg)

    print("Running MPFCM with semi-supervised learning (5% labeled data)...")
    start_time = time.time()
    runtime, cluster_predict, cluster_predict_adj = run_MPFCM(L, c, N, data_new, params, similarity, labels, semi_supervised_ratio=0.05)
    end_time = time.time()
    print(f"MPFCM completed in {runtime:.2f} seconds")

    metrics = {}
    metrics_adj = {}
    combined_features = np.hstack([view for view in data])

    print("\nEvaluating standard clustering (from u):")
    if len(cluster_predict) > 1:
        unique_clusters = np.unique(cluster_predict)
        if len(unique_clusters) > 1:
            try:
                silhouette = silhouette_score(combined_features, cluster_predict)
                metrics["silhouette"] = float(silhouette)
                calinski = calinski_harabasz_score(combined_features, cluster_predict)
                metrics["calinski_harabasz"] = float(calinski)
                davies = davies_bouldin_score(combined_features, cluster_predict)
                metrics["davies_bouldin"] = float(davies)
            except Exception as e:
                print(f"Error calculating unsupervised metrics: {e}")

    print("\nEvaluating adjusted clustering (from u*(1-xi)):")
    if len(cluster_predict_adj) > 1:
        unique_clusters_adj = np.unique(cluster_predict_adj)
        if len(unique_clusters_adj) > 1:
            try:
                silhouette_adj = silhouette_score(combined_features, cluster_predict_adj)
                metrics_adj["silhouette"] = float(silhouette_adj)
                calinski_adj = calinski_harabasz_score(combined_features, cluster_predict_adj)
                metrics_adj["calinski_harabasz"] = float(calinski_adj)
                davies_adj = davies_bouldin_score(combined_features, cluster_predict_adj)
                metrics_adj["davies_bouldin"] = float(davies_adj)
            except Exception as e:
                print(f"Error calculating unsupervised metrics for adjusted clustering: {e}")

    if labels is not None and np.any(labels > 0):
        try:
            valid_indices = labels > 0
            valid_labels = labels[valid_indices]
            valid_clusters = cluster_predict[valid_indices]
            ari = adjusted_rand_score(valid_labels, valid_clusters)
            metrics["ari"] = float(ari)
            nmi = normalized_mutual_info_score(valid_labels, valid_clusters)
            metrics["nmi"] = float(nmi)

            valid_clusters_adj = cluster_predict_adj[valid_indices]
            ari_adj = adjusted_rand_score(valid_labels, valid_clusters_adj)
            metrics_adj["ari"] = float(ari_adj)
            nmi_adj = normalized_mutual_info_score(valid_labels, valid_clusters_adj)
            metrics_adj["nmi"] = float(nmi_adj)
        except Exception as e:
            print(f"Error calculating supervised metrics: {e}")

    return cluster_predict, cluster_predict_adj, metrics, metrics_adj

def main():
    print("Starting MUUFL Full-Image Segmentation using MPFCM")

    # Update this path to your dataset
    filepath = r"/kaggle/working/ssmpfc2/data/muufl_gulfport_campus_1_hsi_220_label.mat"

    data_views, original_shape, rgb_image, hsi_image, lidar_image, mask_2d, labels, pixel_positions = load_muufl_data_full(filepath)

    if data_views is None:
        print("Error loading data. Exiting.")
        return

    c = 11
    L = len(data_views)
    N = len(data_views[0])
    n_anchors = c

    print(f"\nParameters:")
    print(f"Number of views: {L}")
    print(f"Number of clusters: {c}")
    print(f"Number of samples: {N} (after filtering)")
    print(f"Number of anchors: {n_anchors}")

    alpha = 20
    beta = alpha/(100 * 15 *6)
    sigma = 5
    omega = 0.001
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

    height, width = hsi_image.shape[:2]
    final_cluster_image = np.full((height, width), -1, dtype=int)

    print("\nProcessing full image...")
    cluster_predict, cluster_predict_adj, metrics, metrics_adj = process_full_image(data_views, labels, c, n_anchors, params)

    print(f"Final cluster image shape: {final_cluster_image.shape}")
    print(f"Number of pixel positions: {len(pixel_positions)}")
    print(f"Number of clusters predicted: {len(cluster_predict)}")

    for i, (row, col) in enumerate(pixel_positions):
        if row < height and col < width:
            final_cluster_image[row, col] = cluster_predict[i]

    final_cluster_image_std = np.full((height, width), -1, dtype=int)
    final_cluster_image_adj = np.full((height, width), -1, dtype=int)

    for i, (row, col) in enumerate(pixel_positions):
        if row < height and col < width:
            final_cluster_image_std[row, col] = cluster_predict[i]
    for i, (row, col) in enumerate(pixel_positions):
        if row < height and col < width:
            final_cluster_image_adj[row, col] = cluster_predict_adj[i]

    os.makedirs('results', exist_ok=True)

    print(f"\nSegmentation completed!")

if __name__ == "__main__":
    main()


