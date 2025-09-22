import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.io import loadmat
import h5py
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from mpfcm import run_MPFCM
from MV_DAGL import MV_DAGL
from normalization import normalize_multiview_data3


def read_mat(filepath: str):
    try:
        return loadmat(filepath)
    except Exception:
        with h5py.File(filepath, "r") as f:
            out = {k: np.array(v) for k, v in f.items()}
        return out


def resize_multichannel_image(image: np.ndarray, new_size: tuple[int, int], interpolation=cv2.INTER_CUBIC) -> np.ndarray:
    if image.ndim == 3:
        channels = [cv2.resize(image[..., i], new_size, interpolation=interpolation) for i in range(image.shape[-1])]
        return np.stack(channels, axis=-1)
    else:
        return cv2.resize(image, new_size, interpolation=interpolation)


def load_trento_data_resized(
    data_dir: str = "D:/mpfcm_8_6_12 (1)/mpfcm_8_6_12/data/Trento",
    scale: float = 2.0 / 3.0,
):
    """Load Trento HSI + LiDAR + GT, resize by 'scale'.

    Returns
    -------
    data_views : list[np.ndarray]
        [hsi_norm, lidar_norm] each of shape (N, F).
    (height, width) : tuple[int, int]
        Resized dimensions.
    labels_valid : np.ndarray
        1D labels for valid (>0) pixels only.
    pixel_positions_valid : np.ndarray
        Nx2 positions in resized image for valid pixels.
    vis_inputs : dict
        Dict containing resized HSI (for 3-band vis) and resized LiDAR.
    """

    gt_file = os.path.join(data_dir, "GT_Trento.mat")
    hsi_file = os.path.join(data_dir, "HSI_Trento.mat")
    lidar_file = os.path.join(data_dir, "Lidar_Trento.mat")

    # Load GT
    gt_data = read_mat(gt_file)
    gt_key = [k for k in gt_data.keys() if not str(k).startswith("__")][0]
    labels = np.array(gt_data[gt_key]).astype(np.int32)

    # Load HSI
    hsi_data = read_mat(hsi_file)
    hsi_key = [k for k in hsi_data.keys() if not str(k).startswith("__")][0]
    hsi = np.array(hsi_data[hsi_key])  # (H, W, B)

    # Load LiDAR
    lidar_data = read_mat(lidar_file)
    lidar_key = [k for k in lidar_data.keys() if not str(k).startswith("__")][0]
    lidar = np.array(lidar_data[lidar_key])  # (H, W) or (H, W, 1)
    if lidar.ndim == 3 and lidar.shape[-1] == 1:
        lidar = lidar[..., 0]

    # Compute resize dims (2/3 per dimension)
    orig_h, orig_w = hsi.shape[:2]
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    new_size = (new_w, new_h)
    print(f"Resizing Trento to (w={new_w}, h={new_h}) from (w={orig_w}, h={orig_h})")

    # Resize modalities
    resized_hsi = resize_multichannel_image(hsi, new_size, interpolation=cv2.INTER_CUBIC)
    resized_lidar = resize_multichannel_image(lidar, new_size, interpolation=cv2.INTER_CUBIC)
    resized_labels = resize_multichannel_image(labels, new_size, interpolation=cv2.INTER_NEAREST)
    resized_labels = np.round(resized_labels).astype(np.int32)

    # Prepare features
    height, width = new_h, new_w
    hsi_reshaped = resized_hsi.reshape(-1, resized_hsi.shape[-1])
    lidar_reshaped = resized_lidar.reshape(-1, 1)

    # Pixel positions
    pixel_positions = np.indices((height, width)).transpose(1, 2, 0).reshape(-1, 2)

    # Filter valid: label > 0
    labels_flat = resized_labels.flatten()
    valid_indices = labels_flat > 0

    total_points = len(labels_flat)
    valid_points = int(np.sum(valid_indices))
    print(
        f"Filtering labels (label>0): kept {valid_points}/{total_points} "
        f"({valid_points/total_points*100:.1f}%)"
    )

    hsi_reshaped = hsi_reshaped[valid_indices]
    lidar_reshaped = lidar_reshaped[valid_indices]
    labels_valid = labels_flat[valid_indices]
    pixel_positions_valid = pixel_positions[valid_indices]

    # Normalize per view
    hsi_norm = StandardScaler().fit_transform(hsi_reshaped)
    lidar_norm = StandardScaler().fit_transform(lidar_reshaped)

    data_views = [hsi_norm, lidar_norm]
    vis_inputs = {"hsi": resized_hsi, "lidar": resized_lidar}
    return data_views, (height, width), labels_valid, pixel_positions_valid, vis_inputs


def process_full_image(data_views, labels_valid, c, n_anchors, params):
    L = len(data_views)
    N = len(data_views[0])
    print(f"\nProcessing resized Trento image with N={N}, L={L}")

    data = normalize_multiview_data3(data_views)

    # Optional PCA on HSI (slight reduction)
    try:
        hs_idx = 0
        target_dim = min(30, data[hs_idx].shape[1])
        if target_dim < data[hs_idx].shape[1]:
            print(f"Applying PCA on HS to {target_dim} dims...")
            pca = PCA(n_components=target_dim)
            data[hs_idx] = pca.fit_transform(data[hs_idx])
            print(f"HS after PCA: {data[hs_idx].shape}")
    except Exception as e:
        print(f"PCA skipped due to: {e}")

    print("Running MV-DAGL...")
    t0 = time.time()
    z_common, z_list = MV_DAGL(data, n_anchors, max_iter=500, alpha=0.2, tol=1e-6)
    print(f"MV-DAGL done in {time.time()-t0:.2f}s")

    data_new = []
    similarity = []
    for l in range(L):
        tg = np.array(z_list[l].T @ z_list[l])
        similarity.append(tg)
        U, _, _ = svd(z_list[l].T, full_matrices=False)
        data_new.append(U[:, :n_anchors])

    Uc, _, _ = svd(z_common.T, full_matrices=False)
    data_new.append(Uc[:, :n_anchors])
    similarity.append(np.array(z_common.T @ z_common))

    print("Running MPFCM (semi-supervised, 5%)...")
    t0 = time.time()
    runtime, cluster_predict, cluster_predict_adj = run_MPFCM(
        L, c, N, data_new, params, similarity, labels_valid, semi_supervised_ratio=0.05
    )
    print(f"MPFCM done in {runtime:.2f}s (wall {time.time()-t0:.2f}s)")

    metrics = {}
    metrics_adj = {}
    try:
        combined = np.hstack([view for view in data])
        if len(np.unique(cluster_predict)) > 1:
            metrics["silhouette"] = float(silhouette_score(combined, cluster_predict))
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(combined, cluster_predict))
            metrics["davies_bouldin"] = float(davies_bouldin_score(combined, cluster_predict))
        if len(np.unique(cluster_predict_adj)) > 1:
            metrics_adj["silhouette"] = float(silhouette_score(combined, cluster_predict_adj))
            metrics_adj["calinski_harabasz"] = float(calinski_harabasz_score(combined, cluster_predict_adj))
            metrics_adj["davies_bouldin"] = float(davies_bouldin_score(combined, cluster_predict_adj))
    except Exception as e:
        print(f"Unsupervised metrics error: {e}")

    try:
        metrics["ari"] = float(adjusted_rand_score(labels_valid, cluster_predict))
        metrics["nmi"] = float(normalized_mutual_info_score(labels_valid, cluster_predict))
        metrics_adj["ari"] = float(adjusted_rand_score(labels_valid, cluster_predict_adj))
        metrics_adj["nmi"] = float(normalized_mutual_info_score(labels_valid, cluster_predict_adj))
    except Exception as e:
        print(f"Supervised metrics error: {e}")

    return cluster_predict, cluster_predict_adj, metrics, metrics_adj


def visualize_and_save(height, width, pixel_positions, cluster_predict, cluster_predict_adj, vis_inputs):
    os.makedirs("results", exist_ok=True)

    full_std = np.full((height, width), -1, dtype=int)
    full_adj = np.full((height, width), -1, dtype=int)
    for i, (r, c_) in enumerate(pixel_positions):
        full_std[r, c_] = int(cluster_predict[i])
        full_adj[r, c_] = int(cluster_predict_adj[i])

    def colorize(img):
        valid = img >= 0
        u = np.unique(img[valid])
        colors = plt.cm.jet(np.linspace(0, 1, len(u)))
        out = np.zeros((height, width, 3))
        for i, k in enumerate(u):
            out[img == k] = colors[i, :3]
        return out

    std_rgb = colorize(full_std)
    adj_rgb = colorize(full_adj)

    plt.figure(figsize=(16, 10))
    # HSI RGB from first 3 bands if available
    plt.subplot(2, 3, 1)
    hsi = vis_inputs.get("hsi")
    if hsi is not None and hsi.ndim == 3 and hsi.shape[2] >= 3:
        hsi_vis = hsi[:, :, :3]
        hsi_vis = (hsi_vis - hsi_vis.min()) / (hsi_vis.max() - hsi_vis.min() + 1e-8)
        plt.imshow(hsi_vis)
        plt.title("Resized HSI (RGB)")
    else:
        plt.imshow(np.zeros((height, width, 3)))
        plt.title("HSI not available")
    plt.axis("off")

    # LiDAR
    plt.subplot(2, 3, 2)
    lidar = vis_inputs.get("lidar")
    plt.imshow(lidar, cmap="gray")
    plt.title("Resized LiDAR")
    plt.axis("off")

    # Standard
    plt.subplot(2, 3, 4)
    plt.imshow(std_rgb)
    plt.title("Trento Resized - Standard (u)")
    plt.axis("off")

    # Adjusted
    plt.subplot(2, 3, 5)
    plt.imshow(adj_rgb)
    plt.title("Trento Resized - Adjusted (u*(1-ξ))")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("results/trento_mpfcm_full_image_resized_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save individual images and arrays
    np.save("results/trento_mpfcm_labels_full_image_resized.npy", full_std)
    np.save("results/trento_mpfcm_labels_full_image_resized_adj.npy", full_adj)

    plt.figure(figsize=(10, 8))
    plt.imshow(adj_rgb)
    plt.title("Trento Resized - Adjusted (u*(1-ξ))")
    plt.axis("off")
    plt.savefig("results/trento_mpfcm_full_image_resized.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.imshow(std_rgb)
    plt.title("Trento Resized - Standard (u)")
    plt.axis("off")
    plt.savefig("results/trento_mpfcm_full_image_resized_u.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    data_views, (height, width), labels_valid, pixel_positions, vis_inputs = load_trento_data_resized(scale=2.0/3.0)

    # Determine clusters from labels
    unique_classes = np.unique(labels_valid)
    c = int(len(unique_classes))
    print(f"Using c={c} clusters: {unique_classes}")
    n_anchors = 4 * c

    params = {
        "alpha": 20,
        "beta": 20 / (100 *4* 5),
        "sigma": 5,
        "theta": 0.0001,
        "omega": 0.1,
        "rho": 0.0001,
        "EPSILON": 1e-4,
    }

    cluster_predict, cluster_predict_adj, metrics, metrics_adj = process_full_image(
        data_views, labels_valid, c, n_anchors, params
    )

    visualize_and_save(height, width, pixel_positions, cluster_predict, cluster_predict_adj, vis_inputs)

    os.makedirs("results", exist_ok=True)
    import json
    with open("results/trento_full_image_metrics_resized.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open("results/trento_full_image_metrics_resized_adj.json", "w", encoding="utf-8") as f:
        json.dump(metrics_adj, f, indent=2)

    print("\nTrento resized segmentation completed!")
    print("Outputs:")
    print("- results/trento_mpfcm_full_image_resized.png")
    print("- results/trento_mpfcm_full_image_resized_u.png")
    print("- results/trento_mpfcm_full_image_resized_comparison.png")
    print("- results/trento_mpfcm_labels_full_image_resized.npy")
    print("- results/trento_mpfcm_labels_full_image_resized_adj.npy")
    print("- results/trento_full_image_metrics_resized.json")
    print("- results/trento_full_image_metrics_resized_adj.json")


if __name__ == "__main__":
    main()


