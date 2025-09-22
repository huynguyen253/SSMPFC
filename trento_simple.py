import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import json
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, cohen_kappa_score

from mpfcm import run_MPFCM
from MV_DAGL import MV_DAGL
from normalization import normalize_multiview_data3

def load_trento_data():
    """Load Trento dataset"""
    print("Loading Trento data...")
    
    # Load data
    gt_data = sio.loadmat("data/Trento/GT_Trento.mat")
    hsi_data = sio.loadmat("data/Trento/HSI_Trento.mat")
    lidar_data = sio.loadmat("data/Trento/Lidar_Trento.mat")
    
    # Extract data
    labels = gt_data['GT_Trento']
    hsi = hsi_data['HSI_Trento']
    lidar = lidar_data['Lidar_Trento']
    
    # QUAN TRỌNG: Chuyển đổi labels từ uint8 sang int32 để tránh overflow
    labels = labels.astype(np.int32)
    
    print(f"Labels shape: {labels.shape}")
    print(f"Labels dtype: {labels.dtype}")
    print(f"HSI shape: {hsi.shape}")
    print(f"Lidar shape: {lidar.shape}")
    
    # Count valid points (label > 0)
    valid_points = np.sum(labels > 0)
    total_points = labels.size
    print(f"Valid points: {valid_points}/{total_points} ({valid_points/total_points*100:.1f}%)")
    
    # Print label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Label distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  Label {label}: {count} samples ({count/total_points*100:.1f}%)")
    
    return hsi, lidar, labels

def prepare_data(hsi, lidar, labels):
    """Prepare data for processing"""
    height, width = hsi.shape[:2]
    
    # Reshape to 2D
    hsi_2d = hsi.reshape(-1, hsi.shape[-1])
    lidar_2d = lidar.reshape(-1, 1)
    
    # Filter valid points (label > 0)
    valid_mask = labels.flatten() > 0
    hsi_valid = hsi_2d[valid_mask]
    lidar_valid = lidar_2d[valid_mask]
    labels_valid = labels.flatten()[valid_mask]
    
    print(f"Valid samples: {len(labels_valid)}")
    
    # Print distribution of valid labels
    unique_valid_labels, valid_counts = np.unique(labels_valid, return_counts=True)
    print("Valid label distribution:")
    for label, count in zip(unique_valid_labels, valid_counts):
        print(f"  Label {label}: {count} samples ({count/len(labels_valid)*100:.1f}%)")
    
    # Normalize
    scaler_hsi = StandardScaler()
    scaler_lidar = StandardScaler()
    
    hsi_norm = scaler_hsi.fit_transform(hsi_valid)
    lidar_norm = scaler_lidar.fit_transform(lidar_valid)
    
    return [hsi_norm, lidar_norm], labels_valid, valid_mask, (height, width)

def main():
    print("=== TRENTO SEGMENTATION ===")
    
    # Load data
    hsi, lidar, labels = load_trento_data()
    
    # Prepare data
    data_views, labels_valid, valid_mask, (height, width) = prepare_data(hsi, lidar, labels)
    
    # Parameters
    c = len(np.unique(labels_valid))  # Number of clusters
    n_anchors = 4 * c
    L = len(data_views)  # Number of views
    N = len(data_views[0])  # Number of samples
    
    print(f"Clusters: {c}")
    print(f"Anchors: {n_anchors}")
    print(f"Views: {L}")
    print(f"Samples: {N}")
    
    # MPFCM parameters
    params = {
        "alpha": 20,
        "beta": 20/(100 * 5),
        "sigma": 5,
        "theta": 0.001,
        "omega": 0.1,
        "rho": 0.0001,
        "EPSILON": 0.0001,
    }
    
    # Normalize multiview data
    data = normalize_multiview_data3(data_views)
    
    # Run MV-DAGL
    print("Running MV-DAGL...")
    Z_common, Z_specific = MV_DAGL(data, n_anchors, max_iter=100, alpha=0.2, tol=1e-6)
    
    # Prepare data for MPFCM
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
    print("Running MPFCM...")
    runtime, cluster_predict, cluster_predict_adj = run_MPFCM(
        L, c, N, data_new, params, similarity, labels_valid, semi_supervised_ratio=0.05
    )
    
    # Add 1 to match original labels (1-6)
    if cluster_predict is not None:
        cluster_predict += 1
    if cluster_predict_adj is not None:
        cluster_predict_adj += 1
    
    # Evaluate results
    print("\n=== EVALUATION ===")
    
    # Metrics for standard clustering
    if cluster_predict is not None:
        ari = adjusted_rand_score(labels_valid, cluster_predict)
        nmi = normalized_mutual_info_score(labels_valid, cluster_predict)
        kappa = cohen_kappa_score(labels_valid, cluster_predict)
        
        print("Standard clustering (u):")
        print(f"ARI: {ari:.4f}")
        print(f"NMI: {nmi:.4f}")
        print(f"Kappa: {kappa:.4f}")
    
    # Metrics for adjusted clustering
    if cluster_predict_adj is not None:
        ari_adj = adjusted_rand_score(labels_valid, cluster_predict_adj)
        nmi_adj = normalized_mutual_info_score(labels_valid, cluster_predict_adj)
        kappa_adj = cohen_kappa_score(labels_valid, cluster_predict_adj)
        
        print("\nAdjusted clustering (u*(1-xi)):")
        print(f"ARI: {ari_adj:.4f}")
        print(f"NMI: {nmi_adj:.4f}")
        print(f"Kappa: {kappa_adj:.4f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    
    # Create cluster image
    cluster_image = np.zeros((height, width), dtype=int) - 1
    cluster_image_adj = np.zeros((height, width), dtype=int) - 1
    
    # Map back to original positions
    valid_indices = np.where(valid_mask)[0]
    for i, idx in enumerate(valid_indices):
        row, col = np.unravel_index(idx, (height, width))
        if cluster_predict is not None and i < len(cluster_predict):
            cluster_image[row, col] = cluster_predict[i]
        if cluster_predict_adj is not None and i < len(cluster_predict_adj):
            cluster_image_adj[row, col] = cluster_predict_adj[i]
    
    # Save
    np.save('results/trento_clusters.npy', cluster_image)
    np.save('results/trento_clusters_adj.npy', cluster_image_adj)
    
    # Save metrics
    results = {
        'standard': {'ari': ari, 'nmi': nmi, 'kappa': kappa} if cluster_predict is not None else {},
        'adjusted': {'ari': ari_adj, 'nmi': nmi_adj, 'kappa': kappa_adj} if cluster_predict_adj is not None else {}
    }
    
    with open('results/trento_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nResults saved to results/ directory")

if __name__ == "__main__":
    main() 