import os
import time
import numpy as np
from scipy.io import loadmat
import h5py
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, cohen_kappa_score
import matplotlib.pyplot as plt

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

def load_trento_full(data_dir: str = "/kaggle/working/ssmpfc2/data/Trento"):
    gt_file = os.path.join(data_dir, "GT_Trento.mat")
    hsi_file = os.path.join(data_dir, "HSI_Trento.mat")
    lidar_file = os.path.join(data_dir, "Lidar_Trento.mat")

    gt_data = read_mat(gt_file)
    gt_key = [k for k in gt_data.keys() if not str(k).startswith("__")][0]
    labels = np.array(gt_data[gt_key]).astype(np.int32)

    hsi_data = read_mat(hsi_file)
    hsi_key = [k for k in hsi_data.keys() if not str(k).startswith("__")][0]
    hsi = np.array(hsi_data[hsi_key])  # (H, W, B)

    lidar_data = read_mat(lidar_file)
    lidar_key = [k for k in lidar_data.keys() if not str(k).startswith("__")][0]
    lidar = np.array(lidar_data[lidar_key])  # (H, W) or (H, W, 1)
    if lidar.ndim == 3 and lidar.shape[-1] == 1:
        lidar = lidar[..., 0]

    height, width = hsi.shape[:2]

    # Flatten
    hsi_2d = hsi.reshape(-1, hsi.shape[-1])
    lidar_2d = lidar.reshape(-1, 1)

    # Valid label mask (>0)
    labels_flat = labels.flatten()
    valid_mask = labels_flat > 0

    hsi_valid = hsi_2d[valid_mask]
    lidar_valid = lidar_2d[valid_mask]
    labels_valid = labels_flat[valid_mask]

    # Pixel positions for valid points
    rows, cols = np.indices((height, width))
    positions = np.stack([rows.flatten(), cols.flatten()], axis=1)
    pixel_positions_valid = positions[valid_mask]

    # Normalize float32
    hsi_norm = StandardScaler().fit_transform(hsi_valid).astype(np.float32)
    lidar_norm = StandardScaler().fit_transform(lidar_valid).astype(np.float32)

    data_views = [hsi_norm, lidar_norm]
    return data_views, (height, width), labels_valid, pixel_positions_valid, hsi

def process_trento_full(data_views, labels, c, n_anchors, params):
    L = len(data_views)
    N = len(data_views[0])
    print(f"Processing Trento FULL with N={N}, L={L}")

    data = normalize_multiview_data3(data_views)
    data = [np.asarray(v, dtype=np.float32) for v in data]

    # Optional PCA on HSI
    try:
        hs_idx = 0
        target_dim = min(30, data[hs_idx].shape[1])
        if target_dim < data[hs_idx].shape[1]:
            print(f"Applying PCA on HS to {target_dim} dims...")
            pca = PCA(n_components=target_dim)
            data[hs_idx] = pca.fit_transform(data[hs_idx]).astype(np.float32)
            print(f"HS after PCA: {data[hs_idx].shape}")
    except Exception as e:
        print(f"PCA skipped: {e}")

    print("Running MV-DAGL...")
    t0 = time.time()
    z_common, z_list = MV_DAGL(data, n_anchors, max_iter=500, alpha=0.2, tol=1e-6)
    print(f"MV-DAGL done in {time.time()-t0:.2f}s")

    data_new = []
    similarity = []
    for l in range(L):
        Z = np.asarray(z_list[l], dtype=np.float32)
        tg = (Z.T @ Z).astype(np.float32)
        similarity.append(tg)
        U, _, _ = svd(Z.T, full_matrices=False)
        data_new.append(np.asarray(U[:, :n_anchors], dtype=np.float32))

    Zc = np.asarray(z_common, dtype=np.float32)
    Uc, _, _ = svd(Zc.T, full_matrices=False)
    data_new.append(np.asarray(Uc[:, :n_anchors], dtype=np.float32))
    similarity.append((Zc.T @ Zc).astype(np.float32))

    print("Running MPFCM (semi-supervised, 5%)...")
    t0 = time.time()
    runtime, cluster_predict, cluster_predict_adj = run_MPFCM(
        L, c, N, data_new, params, similarity, labels, semi_supervised_ratio=0.05
    )
    print(f"MPFCM done in {runtime:.2f}s (wall {time.time()-t0:.2f}s)")

    # Metrics (optional quick check)
    metrics = {}
    metrics_adj = {}
    try:
        metrics["ari"] = float(adjusted_rand_score(labels, cluster_predict))
        metrics["nmi"] = float(normalized_mutual_info_score(labels, cluster_predict))
        metrics["kappa"] = float(cohen_kappa_score(labels, cluster_predict))
        metrics_adj["ari"] = float(adjusted_rand_score(labels, cluster_predict_adj))
        metrics_adj["nmi"] = float(normalized_mutual_info_score(labels, cluster_predict_adj))
        metrics_adj["kappa"] = float(cohen_kappa_score(labels, cluster_predict_adj))
    except Exception:
        pass

    return cluster_predict, cluster_predict_adj, metrics, metrics_adj

def main():
    print("Starting Trento Image Segmentation")
    data_views, (height, width), labels_valid, pixel_positions, hsi_image = load_trento_full()

    c = int(len(np.unique(labels_valid)))
    n_anchors = 6 * c

    params = {
        "alpha": 20,
        "beta": 20/(100 * 5 * 6),
        "sigma": 5,
        "theta": 0.0001,
        "omega": 0.1,
        "rho": 0.0001,
        "EPSILON": 1e-4,
    }

    cluster_predict, cluster_predict_adj, metrics, metrics_adj = process_trento_full(
        data_views, labels_valid, c, n_anchors, params
    )

    # Map results back to full image
    full_std = np.full((height, width), -1, dtype=int)
    full_adj = np.full((height, width), -1, dtype=int)
    for i, (r, c_) in enumerate(pixel_positions):
        if i < len(cluster_predict):
            full_std[r, c_] = int(cluster_predict[i])
        if i < len(cluster_predict_adj):
            full_adj[r, c_] = int(cluster_predict_adj[i])

    os.makedirs("results", exist_ok=True)
    np.save("results/trento_full_labels.npy", full_std)
    np.save("results/trento_full_labels_adj.npy", full_adj)

    # Visualization: original HSI (first 3 bands), standard u, adjusted u*(1-xi)
    def normalize_rgb_from_hsi(hsi_arr: np.ndarray) -> np.ndarray:
        if hsi_arr.ndim == 3 and hsi_arr.shape[2] >= 3:
            rgb = hsi_arr[:, :, :3].astype(np.float32)
            rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb) + 1e-8)
            return rgb
        # fallback to single channel repeated
        ch = hsi_arr[:, :, 0].astype(np.float32)
        ch = (ch - np.min(ch)) / (np.max(ch) - np.min(ch) + 1e-8)
        return np.stack([ch, ch, ch], axis=-1)

    def colorize_labels(lbl_img: np.ndarray) -> np.ndarray:
        vis = np.zeros((lbl_img.shape[0], lbl_img.shape[1], 3), dtype=np.float32)
        valid = lbl_img >= 0
        unique_vals = np.unique(lbl_img[valid]) if np.any(valid) else []
        cmap = plt.cm.tab20(np.linspace(0, 1, max(len(unique_vals), 1)))
        for i, v in enumerate(unique_vals):
            vis[lbl_img == v] = cmap[i, :3]
        return vis

    orig_rgb = normalize_rgb_from_hsi(hsi_image)
    std_rgb = colorize_labels(full_std)
    adj_rgb = colorize_labels(full_adj)

    # Side-by-side comparison figure
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(orig_rgb)
    plt.title('Trento Original (HSI RGB)')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(std_rgb)
    plt.title('Clustering (u)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(adj_rgb)
    plt.title('Clustering (u*(1-ξ))')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('results/trento_full_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save individual clustering images
    plt.imsave('results/trento_full_u.png', std_rgb)
    plt.imsave('results/trento_full_u_adj.png', adj_rgb)

    # Save metrics
    import json
    with open("results/trento_full_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open("results/trento_full_metrics_adj.json", "w", encoding="utf-8") as f:
        json.dump(metrics_adj, f, indent=2)

if __name__ == "__main__":
    main()


