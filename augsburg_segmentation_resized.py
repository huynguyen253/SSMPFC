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
    confusion_matrix,
    f1_score,
)
from scipy.optimize import linear_sum_assignment
from typing import Optional, Tuple

# Project imports
from mpfcm import run_MPFCM
from MV_DAGL import MV_DAGL
from normalization import normalize_multiview_data3


def read_mat(filepath: str):
    """Robust .mat reader supporting v7.3 via h5py.

    Returns a dict-like of arrays.
    """
    try:
        return loadmat(filepath)
    except Exception:
        # v7.3 files
        with h5py.File(filepath, "r") as f:
            out = {}
            for k, v in f.items():
                out[k] = np.array(v)
            return out


def resize_multichannel_image(image: np.ndarray, new_size: Tuple[int, int], interpolation=cv2.INTER_CUBIC) -> np.ndarray:
    """Resize a 2D/3D image. new_size = (width, height).

    For 3D arrays, resize each channel independently and stack.
    """
    if image.ndim == 3:
        channels = []
        for i in range(image.shape[-1]):
            channels.append(cv2.resize(image[..., i], new_size, interpolation=interpolation))
        return np.stack(channels, axis=-1)
    else:
        return cv2.resize(image, new_size, interpolation=interpolation)


def load_augsburg_data_resized(
    dsm_filepath: str = "D:/Download/Datasets/Datasets/HS-SAR-DSM Augsburg/data_DSM.mat",
    hs_filepath: str = "D:/Download/Datasets/Datasets/HS-SAR-DSM Augsburg/data_HS_LR.mat",
    sar_filepath: str = "D:/Download/Datasets/Datasets/HS-SAR-DSM Augsburg/data_SAR_HR.mat",
    total_filepath: str = "D:/Download/Datasets/Datasets/HS-SAR-DSM Augsburg/TotalImage_transformed.mat",
    scale: float = 0.5,
):
    """Load Augsburg data, resize all modalities and labels by 'scale'.

    Returns
    -------
    data_views : list[np.ndarray]
        List of normalized feature matrices per view, shape (N, F_l) each.
    resized_shapes : tuple[int, int]
        (height, width) of the resized image.
    labels_valid : np.ndarray | None
        1D array of valid labels after filtering (labels != -1), or None if labels missing.
    pixel_positions_valid : np.ndarray
        Nx2 array of (row, col) for the kept pixels (after label filtering).
    resized_labels : np.ndarray
        Resized full GT label grid.
    resized_sar : np.ndarray | None
        Resized SAR image (for visualization), or None.
    """

    print("\nLoading Augsburg data (resized by scale=%.2f)..." % scale)

    # DSM
    dsm_data = read_mat(dsm_filepath)
    if "data_DSM" in dsm_data:
        dsm = dsm_data["data_DSM"]
    else:
        # Fallback/dummy
        dsm = np.random.rand(332, 485)
    print(f"DSM shape (orig): {dsm.shape}")

    # HS
    hs_data = read_mat(hs_filepath)
    if "data_HS_LR" in hs_data:
        hs = hs_data["data_HS_LR"]
    else:
        hs = np.random.rand(332, 485, 180)
    print(f"HS shape (orig): {hs.shape}")

    # SAR
    sar_data = read_mat(sar_filepath)
    if "data_SAR_HR" in sar_data:
        sar = sar_data["data_SAR_HR"]
    else:
        sar = np.random.rand(332, 485, 4)
    print(f"SAR shape (orig): {sar.shape}")

    # Labels (TotalImage)
    total = read_mat(total_filepath)
    labels = None
    for key in total.keys():
        if not str(key).startswith("__"):
            labels = total[key]
            break
    if labels is None:
        labels = np.zeros((332, 485), dtype=int)
    print(f"Labels shape (orig): {labels.shape}")

    # Compute resized size
    orig_h, orig_w = hs.shape[:2]
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    new_size = (new_w, new_h)
    print(f"Resizing to (width={new_w}, height={new_h})")

    # Resize modalities
    resized_hs = resize_multichannel_image(hs, new_size, interpolation=cv2.INTER_CUBIC)
    resized_sar = resize_multichannel_image(sar, new_size, interpolation=cv2.INTER_CUBIC)
    resized_dsm = resize_multichannel_image(dsm, new_size, interpolation=cv2.INTER_CUBIC)

    # Resize labels using nearest-neighbor to preserve integers
    resized_labels = resize_multichannel_image(labels, new_size, interpolation=cv2.INTER_NEAREST)
    resized_labels = np.round(resized_labels).astype(np.int32)

    # Prepare features
    height, width = new_h, new_w
    hs_reshaped = resized_hs.reshape(-1, resized_hs.shape[-1])
    sar_reshaped = resized_sar.reshape(-1, resized_sar.shape[-1])
    dsm_reshaped = resized_dsm.reshape(-1, 1)

    # Pixel positions
    pixel_positions = np.indices((height, width)).transpose(1, 2, 0).reshape(-1, 2)

    # Filter by labels != -1 (keep 0 as valid)
    labels_flat = resized_labels.flatten()
    valid_indices = labels_flat != -1

    total_points = len(labels_flat)
    valid_points = int(np.sum(valid_indices))
    print(
        f"Filtering invalid labels (-1): kept {valid_points}/{total_points} "
        f"({valid_points/total_points*100:.1f}%)"
    )

    hs_reshaped = hs_reshaped[valid_indices]
    sar_reshaped = sar_reshaped[valid_indices]
    dsm_reshaped = dsm_reshaped[valid_indices]
    labels_valid = labels_flat[valid_indices]
    pixel_positions_valid = pixel_positions[valid_indices]

    # Normalize per view
    hs_norm = StandardScaler().fit_transform(hs_reshaped)
    sar_norm = StandardScaler().fit_transform(sar_reshaped)
    dsm_norm = StandardScaler().fit_transform(dsm_reshaped)

    data_views = [hs_norm, sar_norm, dsm_norm]
    return data_views, (height, width), labels_valid, pixel_positions_valid, resized_labels, resized_sar


def cluster_match_to_labels(pred_clusters: np.ndarray, gt_labels: np.ndarray) -> np.ndarray:
    """Map cluster IDs to label IDs using Hungarian on contingency matrix.

    Returns mapped clusters array with same shape as pred_clusters.
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix

    unique_labels = np.unique(gt_labels)
    cm = confusion_matrix(gt_labels, pred_clusters, labels=unique_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col_ind[i]: unique_labels[i] for i in range(len(row_ind))}
    return np.array([mapping.get(c, -1) for c in pred_clusters])


def process_full_image(data_views, labels_valid, c, n_anchors, params):
    """Run MV-DAGL + MPFCM over full resized image and compute metrics."""
    L = len(data_views)
    N = len(data_views[0])
    print(f"\nProcessing resized Augsburg image with N={N}, L={L}")

    # Normalize multiview data (project helper)
    data = normalize_multiview_data3(data_views)

    # Optional PCA on HS to reduce dimension slightly (robustness/perf)
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

    # MV-DAGL
    print("Running MV-DAGL...")
    t0 = time.time()
    z_common, z_list = MV_DAGL(data, n_anchors, max_iter=1000, alpha=0.2, tol=1e-6)
    print(f"MV-DAGL done in {time.time()-t0:.2f}s")

    # Prepare views for MPFCM
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

    # MPFCM (semi-supervised: 5%)
    print("Running MPFCM (semi-supervised, 5%)...")
    t0 = time.time()
    runtime, cluster_predict, cluster_predict_adj = run_MPFCM(
        L, c, N, data_new, params, similarity, labels_valid, semi_supervised_ratio=0.05
    )
    print(f"MPFCM done in {runtime:.2f}s (wall {time.time()-t0:.2f}s)")

    # Evaluate metrics
    metrics = {}
    metrics_adj = {}

    try:
        combined_features = np.hstack([view for view in data])
        # Standard
        if len(np.unique(cluster_predict)) > 1:
            metrics["silhouette"] = float(silhouette_score(combined_features, cluster_predict))
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(combined_features, cluster_predict))
            metrics["davies_bouldin"] = float(davies_bouldin_score(combined_features, cluster_predict))
        # Adjusted
        if len(np.unique(cluster_predict_adj)) > 1:
            metrics_adj["silhouette"] = float(silhouette_score(combined_features, cluster_predict_adj))
            metrics_adj["calinski_harabasz"] = float(calinski_harabasz_score(combined_features, cluster_predict_adj))
            metrics_adj["davies_bouldin"] = float(davies_bouldin_score(combined_features, cluster_predict_adj))
    except Exception as e:
        print(f"Unsupervised metrics error: {e}")

    # Supervised (valid labels only)
    try:
        # Compare directly (label-invariant)
        metrics["ari"] = float(adjusted_rand_score(labels_valid, cluster_predict))
        metrics["nmi"] = float(normalized_mutual_info_score(labels_valid, cluster_predict))

        metrics_adj["ari"] = float(adjusted_rand_score(labels_valid, cluster_predict_adj))
        metrics_adj["nmi"] = float(normalized_mutual_info_score(labels_valid, cluster_predict_adj))
    except Exception as e:
        print(f"Supervised metrics error: {e}")

    return cluster_predict, cluster_predict_adj, metrics, metrics_adj


def visualize_and_save(
    height: int,
    width: int,
    pixel_positions: np.ndarray,
    cluster_predict: np.ndarray,
    cluster_predict_adj: np.ndarray,
    resized_sar: Optional[np.ndarray],
    resized_labels: np.ndarray,
    c: int,
):
    os.makedirs("results", exist_ok=True)

    # Build full images
    full_std = np.full((height, width), -1, dtype=int)
    full_adj = np.full((height, width), -1, dtype=int)
    for i, (r, c_) in enumerate(pixel_positions):
        full_std[r, c_] = int(cluster_predict[i])
        full_adj[r, c_] = int(cluster_predict_adj[i])

    # Helper to colorize
    def colorize(cluster_image: np.ndarray) -> np.ndarray:
        valid = cluster_image >= 0
        unique_clusters = np.unique(cluster_image[valid])
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_clusters)))
        cluster_to_color = {int(k): colors[i, :3] for i, k in enumerate(unique_clusters)}
        color_img = np.zeros((height, width, 3), dtype=float)
        for k, col in cluster_to_color.items():
            color_img[cluster_image == k] = col
        return color_img

    # Build GT color palette from labels (>0)
    gt_valid = resized_labels > 0
    gt_classes = np.unique(resized_labels[gt_valid])
    gt_colors = plt.cm.jet(np.linspace(0, 1, max(1, len(gt_classes))))
    gt_palette = {int(k): gt_colors[i, :3] for i, k in enumerate(gt_classes)}

    def colorize_by_gt_palette(pred_image: np.ndarray) -> np.ndarray:
        # Map prediction clusters to GT classes by maximizing overlap
        mask = (pred_image >= 0) & (resized_labels > 0)
        pred_vals = np.unique(pred_image[mask])
        if pred_vals.size == 0 or gt_classes.size == 0:
            return colorize(pred_image)
        contingency = np.zeros((len(pred_vals), len(gt_classes)), dtype=np.int64)
        for i, p in enumerate(pred_vals):
            pm = pred_image[mask] == p
            for j, g in enumerate(gt_classes):
                contingency[i, j] = np.sum(pm & (resized_labels[mask] == g))
        r_idx, c_idx = linear_sum_assignment(-contingency)
        pred_to_gt = {}
        for i, j in zip(r_idx, c_idx):
            pred_to_gt[int(pred_vals[i])] = int(gt_classes[j])
        # Colorize using GT colors
        rgb = np.zeros((height, width, 3), dtype=float)
        for p_val in pred_vals:
            gt_lab = pred_to_gt.get(int(p_val))
            if gt_lab is not None:
                rgb[pred_image == p_val] = gt_palette[gt_lab]
        return rgb

    std_rgb = colorize_by_gt_palette(full_std)
    adj_rgb = colorize_by_gt_palette(full_adj)

    # Compute per-class F1 and confusion matrices (standard and adjusted) using Hungarian mapping
    try:
        # 1D GT labels at valid positions
        gt_valid_1d = np.array([resized_labels[r, c_] for r, c_ in pixel_positions])
        true_labels_unique = np.unique(gt_valid_1d)

        # Helper: map predictions to GT via Hungarian
        def map_pred_to_gt(pred_1d: np.ndarray) -> np.ndarray:
            pred_unique = np.unique(pred_1d)
            contingency = np.zeros((len(true_labels_unique), len(pred_unique)), dtype=np.int64)
            true_index = {lbl: i for i, lbl in enumerate(true_labels_unique)}
            pred_index = {lbl: j for j, lbl in enumerate(pred_unique)}
            for t, p in zip(gt_valid_1d, pred_1d):
                contingency[true_index[t], pred_index[p]] += 1
            r_idx, c_idx = linear_sum_assignment(contingency.max() - contingency)
            mapping = {pred_unique[j]: true_labels_unique[i] for i, j in zip(r_idx, c_idx)}
            unmatched = set(range(len(pred_unique))) - set(c_idx)
            for j in unmatched:
                best_i = int(np.argmax(contingency[:, j]))
                mapping[pred_unique[j]] = true_labels_unique[best_i]
            return np.array([mapping[p] for p in pred_1d])

        mapped_std_1d = map_pred_to_gt(cluster_predict.astype(int))
        mapped_adj_1d = map_pred_to_gt(cluster_predict_adj.astype(int))

        # Build 2D mapped images
        mapped_std_image = np.full((height, width), -1, dtype=int)
        mapped_adj_image = np.full((height, width), -1, dtype=int)
        for i, (r, c_) in enumerate(pixel_positions):
            mapped_std_image[r, c_] = int(mapped_std_1d[i])
            mapped_adj_image[r, c_] = int(mapped_adj_1d[i])

        # Per-class F1 and confusion matrices
        ordered_labels = list(np.sort(true_labels_unique))
        cm_std = confusion_matrix(gt_valid_1d, mapped_std_1d, labels=ordered_labels)
        cm_adj = confusion_matrix(gt_valid_1d, mapped_adj_1d, labels=ordered_labels)
        f1_std = f1_score(gt_valid_1d, mapped_std_1d, labels=ordered_labels, average=None, zero_division=0)
        f1_adj = f1_score(gt_valid_1d, mapped_adj_1d, labels=ordered_labels, average=None, zero_division=0)

        # Save JSONs
        import json
        with open('results/augsburg_confusion_matrix_standard.json', 'w', encoding='utf-8') as f:
            json.dump({"labels": [int(x) for x in ordered_labels], "matrix": cm_std.tolist()}, f, indent=2)
        with open('results/augsburg_confusion_matrix_adjusted.json', 'w', encoding='utf-8') as f:
            json.dump({"labels": [int(x) for x in ordered_labels], "matrix": cm_adj.tolist()}, f, indent=2)
        with open('results/augsburg_per_class_f1_standard.json', 'w', encoding='utf-8') as f:
            json.dump({str(int(lbl)): float(val) for lbl, val in zip(ordered_labels, f1_std)}, f, indent=2)
        with open('results/augsburg_per_class_f1_adjusted.json', 'w', encoding='utf-8') as f:
            json.dump({str(int(lbl)): float(val) for lbl, val in zip(ordered_labels, f1_adj)}, f, indent=2)

        # Binary disagreement heatmap (adjusted vs GT)
        valid_mask = resized_labels != -1
        disagreement_mask_truth = ((mapped_adj_image != resized_labels) & valid_mask).astype(np.uint8)
        valid_pixels = int(np.sum(valid_mask))
        disagreement_pixels = int(np.sum(disagreement_mask_truth))
        ratio = (disagreement_pixels / valid_pixels) if valid_pixels > 0 else 0.0
        print(f"Augsburg GT disagreement (adjusted vs GT): {disagreement_pixels}/{valid_pixels} ({ratio*100:.2f}%)")

        # Save heatmap image and array
        plt.figure(figsize=(8, 6))
        plt.imshow(disagreement_mask_truth, cmap='gray', vmin=0, vmax=1)
        plt.colorbar(label='Disagreement (1=diff, 0=same)')
        plt.title('Binary Disagreement Heatmap (u*(1-ξ) vs Ground Truth)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('results/augsburg_disagreement_heatmap_truth_binary.png', dpi=300, bbox_inches='tight')
        plt.close()
        np.save('results/augsburg_disagreement_mask_truth_binary.npy', disagreement_mask_truth)

        # Short logs
        print("Saved:")
        print("- results/augsburg_confusion_matrix_standard.json")
        print("- results/augsburg_confusion_matrix_adjusted.json")
        print("- results/augsburg_per_class_f1_standard.json")
        print("- results/augsburg_per_class_f1_adjusted.json")
        print("- results/augsburg_disagreement_heatmap_truth_binary.png")
        print("- results/augsburg_disagreement_mask_truth_binary.npy")
    except Exception as e:
        print(f"Error computing per-class F1/confusion or disagreement heatmap: {e}")

    # Base figure
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    if resized_sar is not None and resized_sar.shape[2] >= 3:
        sar_vis = resized_sar[:, :, :3]
        sar_vis = (sar_vis - sar_vis.min()) / (sar_vis.max() - sar_vis.min() + 1e-8)
        plt.imshow(sar_vis)
        plt.title("Resized SAR (3-ch)")
    else:
        plt.imshow(np.zeros((height, width, 3)))
        plt.title("SAR not available")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(std_rgb)
    plt.title("Augsburg Resized - Standard Clusters (u)")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(adj_rgb)
    plt.title("Augsburg Resized - Adjusted Clusters (u*(1-ξ))")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    if resized_sar is not None and resized_sar.shape[2] >= 3:
        overlay = 0.7 * adj_rgb
        sar_vis = resized_sar[:, :, :3]
        sar_vis = (sar_vis - sar_vis.min()) / (sar_vis.max() - sar_vis.min() + 1e-8)
        overlay += 0.3 * sar_vis
        overlay = np.clip(overlay, 0, 1)
        plt.imshow(overlay)
        plt.title("Adjusted Overlay on SAR")
        plt.axis("off")
    else:
        plt.imshow(adj_rgb)
        plt.title("Adjusted (no SAR overlay)")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("results/augsburg_mpfcm_full_image_resized_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save arrays
    np.save("results/augsburg_mpfcm_labels_full_image_resized.npy", full_std)
    np.save("results/augsburg_mpfcm_labels_full_image_resized_adj.npy", full_adj)

    # Also save a single image for adjusted result
    plt.figure(figsize=(10, 8))
    plt.imshow(adj_rgb)
    plt.title("Augsburg Resized - Adjusted Clusters")
    plt.axis("off")
    plt.savefig("results/augsburg_mpfcm_full_image_resized.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save a single image for standard (u) result
    plt.figure(figsize=(10, 8))
    plt.imshow(std_rgb)
    plt.title("Augsburg Resized - Standard Clusters (u)")
    plt.axis("off")
    plt.savefig("results/augsburg_mpfcm_full_image_resized_u.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    # Load + resize data (1/2 per user request)
    data_views, (height, width), labels_valid, pixel_positions, resized_labels, resized_sar = load_augsburg_data_resized(scale=0.5)

    # Determine number of clusters from labels > 0; ensure expected 7 classes
    positive_labels = labels_valid[labels_valid > 0]
    unique_pos = np.unique(positive_labels) if positive_labels.size > 0 else []
    c = 7
    n_anchors = 1 * c

    # MPFCM params (aligned with Augsburg pipeline)
    params = {
        "alpha": 20,
        "beta": 20 / (100 *6* 5),
        "sigma": 5,
        "theta": 0.0001,
        "omega": 1,
        "rho": 0.0001,
        "EPSILON": 1e-4,
    }

    cluster_predict, cluster_predict_adj, metrics, metrics_adj = process_full_image(
        data_views, labels_valid, c, n_anchors, params
    )

    # Visualize and save
    visualize_and_save(height, width, pixel_positions, cluster_predict, cluster_predict_adj, resized_sar, resized_labels, c)

    # Save metrics
    os.makedirs("results", exist_ok=True)
    import json

    with open("results/augsburg_full_image_metrics_resized.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open("results/augsburg_full_image_metrics_resized_adj.json", "w", encoding="utf-8") as f:
        json.dump(metrics_adj, f, indent=2)

    print("\nAugsburg resized segmentation completed!")
    print("Outputs:")
    print("- results/augsburg_mpfcm_full_image_resized.png")
    print("- results/augsburg_mpfcm_full_image_resized_comparison.png")
    print("- results/augsburg_mpfcm_labels_full_image_resized.npy")
    print("- results/augsburg_mpfcm_labels_full_image_resized_adj.npy")
    print("- results/augsburg_full_image_metrics_resized.json")
    print("- results/augsburg_full_image_metrics_resized_adj.json")


if __name__ == "__main__":
    main()


