# Image Segmentation with Multi-part Division
import numpy as np
import os
import scipy.io as sio
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time
import random
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import math
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
        # First try normal scipy.io.loadmat
        data = loadmat(filepath)
        return data
    except Exception as e:
        # If that fails, try h5py approach for newer MAT files
        try:
            f = h5py.File(filepath, 'r')
            data = {}
            for k, v in f.items():
                data[k] = np.array(v)
            return data
        except Exception as e2:
            print(f"Error loading MAT file with h5py: {e2}")
            # Create dummy data
            return {"dummy_data": np.zeros((332, 485))}

def save_part_data(part_name, hs_data, sar_data, dsm_data, labels, training_mask, test_mask, valid_points_mask, indices):
    """
    Save the data for a specific image part to disk for future use.
    
    Args:
        part_name: Name of the image part
        hs_data: Hyperspectral data for the part
        sar_data: SAR data for the part
        dsm_data: DSM data for the part
        labels: Ground truth labels for the part
        training_mask: Training mask for the part
        test_mask: Test mask for the part
        valid_points_mask: Valid points mask for the part
        indices: Indices of the part in the original image (start_h, end_h, start_w, end_w)
    """
    # Create directory if it doesn't exist
    os.makedirs('parts_augsburg', exist_ok=True)
    
    # Save all data components
    np.save(f'parts_augsburg/{part_name}_hs.npy', hs_data)
    np.save(f'parts_augsburg/{part_name}_sar.npy', sar_data)
    np.save(f'parts_augsburg/{part_name}_dsm.npy', dsm_data)
    
    # Save masks and labels if they exist
    if labels is not None:
        np.save(f'parts_augsburg/{part_name}_labels.npy', labels)
    
    if training_mask is not None:
        np.save(f'parts_augsburg/{part_name}_training_mask.npy', training_mask)
    
    if test_mask is not None:
        np.save(f'parts_augsburg/{part_name}_test_mask.npy', test_mask)
    
    if valid_points_mask is not None:
        np.save(f'parts_augsburg/{part_name}_valid_points_mask.npy', valid_points_mask)
    
    # Save indices
    np.save(f'parts_augsburg/{part_name}_indices.npy', indices)
    
    print(f"Saved data for part {part_name} to disk")

def load_part_data(part_name):
    """
    Load the data for a specific image part from disk.
    
    Args:
        part_name: Name of the image part
        
    Returns:
        hs_data, sar_data, dsm_data, labels, training_mask, test_mask, valid_points_mask
        Returns None for any component that doesn't exist
    """
    try:
        # Load main data components
        hs_data = np.load(f'parts_augsburg/{part_name}_hs.npy')
        sar_data = np.load(f'parts_augsburg/{part_name}_sar.npy')
        dsm_data = np.load(f'parts_augsburg/{part_name}_dsm.npy')
        
        # Try to load masks and labels, set to None if they don't exist
        try:
            labels = np.load(f'parts_augsburg/{part_name}_labels.npy')
        except:
            labels = None
        
        try:
            training_mask = np.load(f'parts_augsburg/{part_name}_training_mask.npy')
        except:
            training_mask = None
        
        try:
            test_mask = np.load(f'parts_augsburg/{part_name}_test_mask.npy')
        except:
            test_mask = None
        
        try:
            valid_points_mask = np.load(f'parts_augsburg/{part_name}_valid_points_mask.npy')
        except:
            valid_points_mask = None
        
        return hs_data, sar_data, dsm_data, labels, training_mask, test_mask, valid_points_mask
    
    except Exception as e:
        print(f"Error loading data for part {part_name}: {e}")
        return None, None, None, None, None, None, None

def visualize_part_clustering(part_name, image_shape, cluster_predictions, pixel_positions, save_dir='parts_augsburg_clusters'):
    """
    Visualize clustering results for a specific part and save to disk
    
    Args:
        part_name: Name of the image part
        image_shape: Shape of the image (height, width)
        cluster_predictions: Cluster predictions for each valid point
        pixel_positions: Positions of valid points in the image
        save_dir: Directory to save visualization
        
    Returns:
        cluster_image: Image with cluster assignments
    """
    # Create empty image for clusters
    height, width = image_shape
    cluster_image = np.zeros((height, width), dtype=int) - 1  # -1 means no cluster assigned
    
    # Assign cluster predictions to their positions
    for i, (row, col) in enumerate(pixel_positions):
        if i < len(cluster_predictions):
            cluster_image[row, col] = cluster_predictions[i]
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate colormap for visualization
    # Get number of unique clusters
    unique_clusters = np.unique(cluster_predictions)
    num_clusters = len(unique_clusters)
    
    # Create a colormap
    colors = plt.cm.jet(np.linspace(0, 1, num_clusters))
    
    # Create a colored version for visualization
    colored_image = np.zeros((height, width, 3))
    
    # Map cluster indices to colors
    for i, cluster in enumerate(unique_clusters):
        mask = cluster_image == cluster
        colored_image[mask] = colors[i, :3]
    
    # Save as image
    plt.figure(figsize=(8, 8))
    plt.imshow(colored_image)
    plt.title(f'Clusters for {part_name}')
    plt.axis('off')
    plt.savefig(f'{save_dir}/{part_name}_clusters.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save cluster assignments
    np.save(f'{save_dir}/{part_name}_clusters.npy', cluster_image)
    
    return cluster_image

def visualize_part_results(full_shape, hs_data, sar_data, dsm_data, parts_results, n_rows, n_cols):
    """
    Combine all part results into a full image and visualize
    
    Args:
        full_shape: Shape of the full image (height, width)
        hs_data: Full hyperspectral data
        sar_data: Full SAR data
        dsm_data: Full DSM data
        parts_results: List of dictionaries with results for each part
        n_rows: Number of rows in the grid
        n_cols: Number of columns in the grid
        
    Returns:
        full_cluster_image: Full image with cluster assignments
    """
    height, width = full_shape
    
    # Create empty image for full clustering result
    full_cluster_image = np.zeros((height, width), dtype=int) - 1  # -1 means no cluster assigned
    
    # Fill in the full cluster image with results from each part
    for part_result in parts_results:
        indices = part_result['indices']
        pixel_positions = part_result['pixel_positions']
        cluster_predict = part_result['cluster_predict']
        
        # Calculate global positions
        start_h, end_h, start_w, end_w = indices
        
        # Assign cluster predictions to their global positions
        for i, (local_row, local_col) in enumerate(pixel_positions):
            global_row = start_h + local_row
            global_col = start_w + local_col
            
            if i < len(cluster_predict) and 0 <= global_row < height and 0 <= global_col < width:
                full_cluster_image[global_row, global_col] = cluster_predict[i]
    
    # Save full cluster image
    os.makedirs('results', exist_ok=True)
    np.save(f'results/augsburg_mpfcm_labels_parts_{n_rows}x{n_cols}.npy', full_cluster_image)
    
    # Generate colormap for visualization
    # Get number of unique clusters
    unique_clusters = np.unique(full_cluster_image[full_cluster_image >= 0])
    num_clusters = len(unique_clusters)
    
    # Create a colormap
    colors = plt.cm.jet(np.linspace(0, 1, num_clusters))
    
    # Create a colored version for visualization
    colored_image = np.zeros((height, width, 3))
    
    # Map cluster indices to colors
    cluster_to_color_idx = {cluster: i for i, cluster in enumerate(unique_clusters)}
    
    # Set colors for each valid point
    for i in range(height):
        for j in range(width):
            cluster = full_cluster_image[i, j]
            if cluster >= 0:
                color_idx = cluster_to_color_idx[cluster]
                colored_image[i, j] = colors[color_idx, :3]
    
    # Save as image
    plt.figure(figsize=(10, 10))
    plt.imshow(colored_image)
    plt.title(f'Combined clustering result ({n_rows}x{n_cols} parts)')
    plt.axis('off')
    plt.savefig(f'results/augsburg_mpfcm_parts_{n_rows}x{n_cols}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create an overlay version with SAR data as background
    if sar_data is not None and sar_data.shape[2] >= 3:
        sar_vis = sar_data[:, :, :3].copy()
        # Normalize
        sar_vis = (sar_vis - np.min(sar_vis)) / (np.max(sar_vis) - np.min(sar_vis))
        
        # Create overlay image (50% clustering, 50% SAR)
        overlay_image = np.zeros((height, width, 3))
        for i in range(height):
            for j in range(width):
                cluster = full_cluster_image[i, j]
                if cluster >= 0:
                    color_idx = cluster_to_color_idx[cluster]
                    overlay_image[i, j] = 0.7 * colors[color_idx, :3] + 0.3 * sar_vis[i, j]
                else:
                    overlay_image[i, j] = sar_vis[i, j]
        
        # Save overlay image
        plt.figure(figsize=(10, 10))
        plt.imshow(overlay_image)
        plt.title('Clustering result overlaid on SAR data')
        plt.axis('off')
        plt.savefig(f'results/augsburg_mpfcm_overlay_{n_rows}x{n_cols}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return full_cluster_image

def evaluate_overall_clustering(hs_data, ground_truth, parts_results, n_rows, n_cols, valid_mask=None):
    """
    Evaluate the overall clustering performance across all parts
    
    Args:
        hs_data: Full hyperspectral data
        ground_truth: Full ground truth labels (nhãn 0-7, với 0 là không có ground truth)
        parts_results: List of dictionaries with results for each part
        n_rows: Number of rows in the grid
        n_cols: Number of columns in the grid
        valid_mask: Mask of valid points to evaluate (points with ground truth > 0)
        
    Returns:
        metrics_adj: Dictionary with evaluation metrics for adjusted clustering (u*(1-xi))
        metrics: Dictionary with evaluation metrics for standard clustering (u)
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.metrics import cohen_kappa_score
    from sklearn.metrics import confusion_matrix
    from scipy.optimize import linear_sum_assignment
    
    print("\n" + "="*50)
    print("EVALUATING OVERALL CLUSTERING PERFORMANCE")
    print("="*50)
    
    # Get dimensions
    height, width = hs_data.shape[:2]
    
    # Create empty images for full clustering result
    full_cluster_image = np.zeros((height, width), dtype=int) - 1
    full_cluster_image_adj = np.zeros((height, width), dtype=int) - 1
    
    # Fill in the full cluster images with results from each part
    for part_result in parts_results:
        indices = part_result['indices']
        pixel_positions = part_result['pixel_positions']
        cluster_predict = part_result['cluster_predict']
        cluster_predict_adj = part_result['cluster_predict_adj']
        
        # Calculate global positions
        start_h, end_h, start_w, end_w = indices
        
        # Assign cluster predictions to their global positions
        for i, (local_row, local_col) in enumerate(pixel_positions):
            global_row = start_h + local_row
            global_col = start_w + local_col
            
            if i < len(cluster_predict) and 0 <= global_row < height and 0 <= global_col < width:
                full_cluster_image[global_row, global_col] = cluster_predict[i]
                if cluster_predict_adj is not None and i < len(cluster_predict_adj):
                    full_cluster_image_adj[global_row, global_col] = cluster_predict_adj[i]
    
    # Create masks for evaluation
    if valid_mask is not None:
        # Use only points with valid ground truth
        eval_mask = valid_mask
    else:
        # Use points where we have cluster predictions
        eval_mask = (full_cluster_image >= 0)
    
    # Filter out points with no ground truth
    eval_mask = eval_mask & (ground_truth >= 0)
    
    # Get ground truth and predictions for evaluation
    gt_flat = ground_truth[eval_mask]
    pred_flat = full_cluster_image[eval_mask]
    pred_adj_flat = full_cluster_image_adj[eval_mask] if np.any(full_cluster_image_adj >= 0) else None
    
    # Check if we have enough data for evaluation
    if len(gt_flat) < 2:
        print("Not enough valid points with ground truth for evaluation")
        return None, None
    
    print(f"Evaluating on {len(gt_flat)} valid points with ground truth")
    
    # We need to map cluster IDs to ground truth class IDs
    # This is because clustering algorithms produce arbitrary labels
    def cluster_match(clusters, labels):
        # Create contingency matrix
        contingency = confusion_matrix(labels, clusters, labels=np.unique(labels))
        
        # Find optimal assignment
        row_ind, col_ind = linear_sum_assignment(-contingency)
        
        # Create mapping dictionary
        mapping = {col_ind[i]: np.unique(labels)[i] for i in range(len(row_ind))}
        
        # Apply mapping to clusters
        mapped_clusters = np.array([mapping.get(c, -1) for c in clusters])
        
        return mapped_clusters
    
    # Map cluster IDs to ground truth class IDs
    print("\nMatching clusters to ground truth classes...")
    pred_mapped = cluster_match(pred_flat, gt_flat)
    
    # Calculate evaluation metrics
    metrics = {}
    
    # Classification metrics
    accuracy = accuracy_score(gt_flat, pred_mapped)
    f1 = f1_score(gt_flat, pred_mapped, average='weighted')
    precision = precision_score(gt_flat, pred_mapped, average='weighted', zero_division=0)
    recall = recall_score(gt_flat, pred_mapped, average='weighted', zero_division=0)
    
    # Clustering metrics
    ari = adjusted_rand_score(gt_flat, pred_flat)
    nmi = normalized_mutual_info_score(gt_flat, pred_flat)
    # Thêm kappa metric
    kappa = cohen_kappa_score(gt_flat, pred_mapped)
    
    # Store metrics
    metrics['accuracy'] = accuracy
    metrics['f1_score'] = f1
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['adjusted_rand_index'] = ari
    metrics['normalized_mutual_info'] = nmi
    metrics['kappa'] = kappa
    
    # Create confusion matrix
    cm = confusion_matrix(gt_flat, pred_mapped)
    metrics['confusion_matrix'] = cm
    
    # Save metrics to disk
    os.makedirs('results', exist_ok=True)
    np.save('results/augsburg_metrics.npy', metrics)
    
    # Print metrics
    print("\nEvaluation of standard clustering (from u):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Normalized Mutual Information: {nmi:.4f}")
    print(f"Kappa Coefficient: {kappa:.4f}")
    
    # Print confusion matrix summary
    print("\nConfusion Matrix Summary:")
    unique_classes = np.unique(gt_flat)
    for i, cls in enumerate(unique_classes):
        total = np.sum(gt_flat == cls)
        correct = cm[i, i]
        print(f"Class {cls}: {correct}/{total} correct ({100*correct/total:.2f}%)")
    
    # Also evaluate adjusted clustering if available
    metrics_adj = None
    if pred_adj_flat is not None and len(pred_adj_flat) > 0:
        print("\nEvaluating adjusted clustering (from u*(1-xi))...")
        
        # Map adjusted cluster IDs to ground truth class IDs
        pred_adj_mapped = cluster_match(pred_adj_flat, gt_flat)
        
        # Calculate evaluation metrics
        metrics_adj = {}
        
        # Classification metrics
        accuracy_adj = accuracy_score(gt_flat, pred_adj_mapped)
        f1_adj = f1_score(gt_flat, pred_adj_mapped, average='weighted')
        precision_adj = precision_score(gt_flat, pred_adj_mapped, average='weighted', zero_division=0)
        recall_adj = recall_score(gt_flat, pred_adj_mapped, average='weighted', zero_division=0)
        
        # Clustering metrics
        ari_adj = adjusted_rand_score(gt_flat, pred_adj_flat)
        nmi_adj = normalized_mutual_info_score(gt_flat, pred_adj_flat)
        # Thêm kappa metric
        kappa_adj = cohen_kappa_score(gt_flat, pred_adj_mapped)
        
        # Store metrics
        metrics_adj['accuracy'] = accuracy_adj
        metrics_adj['f1_score'] = f1_adj
        metrics_adj['precision'] = precision_adj
        metrics_adj['recall'] = recall_adj
        metrics_adj['adjusted_rand_index'] = ari_adj
        metrics_adj['normalized_mutual_info'] = nmi_adj
        metrics_adj['kappa'] = kappa_adj
        
        # Create confusion matrix
        cm_adj = confusion_matrix(gt_flat, pred_adj_mapped)
        metrics_adj['confusion_matrix'] = cm_adj
        
        # Save metrics to disk
        np.save('results/augsburg_metrics_adj.npy', metrics_adj)
        
        # Print metrics
        print("\nEvaluation of adjusted clustering (from u*(1-xi)):")
        print(f"Accuracy: {accuracy_adj:.4f}")
        print(f"F1-score (weighted): {f1_adj:.4f}")
        print(f"Precision (weighted): {precision_adj:.4f}")
        print(f"Recall (weighted): {recall_adj:.4f}")
        print(f"Adjusted Rand Index: {ari_adj:.4f}")
        print(f"Normalized Mutual Information: {nmi_adj:.4f}")
        print(f"Kappa Coefficient: {kappa_adj:.4f}")
        
        # Print confusion matrix summary
        print("\nConfusion Matrix Summary (Adjusted):")
        for i, cls in enumerate(unique_classes):
            total = np.sum(gt_flat == cls)
            correct = cm_adj[i, i]
            print(f"Class {cls}: {correct}/{total} correct ({100*correct/total:.2f}%)")
    
    # Save detailed results for future analysis
    result_dict = {
        'ground_truth': gt_flat,
        'predictions': pred_flat,
        'predictions_mapped': pred_mapped,
        'metrics': metrics
    }
    if pred_adj_flat is not None:
        result_dict['predictions_adj'] = pred_adj_flat
        result_dict['predictions_adj_mapped'] = pred_adj_mapped if 'pred_adj_mapped' in locals() else None
        result_dict['metrics_adj'] = metrics_adj
    
    np.save('results/augsburg_detailed_results.npy', result_dict)
    
    # Create and save confusion matrix visualizations
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Set tick labels
    unique_classes = np.unique(gt_flat)
    tick_marks = np.arange(len(unique_classes))
    plt.xticks(tick_marks, unique_classes)
    plt.yticks(tick_marks, unique_classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('results/confusion_matrix.png', dpi=150)
    plt.close()
    
    # Also visualize the adjusted confusion matrix if available
    if 'cm_adj' in locals():
        plt.figure(figsize=(10, 8))
        plt.imshow(cm_adj, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix (Adjusted)')
        plt.colorbar()
        
        # Set tick labels
        plt.xticks(tick_marks, unique_classes)
        plt.yticks(tick_marks, unique_classes)
        
        # Add text annotations
        thresh = cm_adj.max() / 2.
        for i in range(cm_adj.shape[0]):
            for j in range(cm_adj.shape[1]):
                plt.text(j, i, format(cm_adj[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm_adj[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('results/confusion_matrix_adj.png', dpi=150)
        plt.close()
    
    # Trả về metrics_adj trước để ưu tiên kết quả từ u*(1-xi)
    return metrics_adj, metrics

# Thêm các import cần thiết
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import confusion_matrix

from mpfcm import main_MPFCM, run_MPFCM
from MV_DAGL import MV_DAGL

from normalization import normalize_multiview_data3

def load_augsburg_data(dsm_filepath="D:/Download/Datasets/Datasets/HS-SAR-DSM Augsburg/data_DSM.mat", 
                        hs_filepath="D:/Download/Datasets/Datasets/HS-SAR-DSM Augsburg/data_HS_LR.mat", 
                        sar_filepath="D:/Download/Datasets/Datasets/HS-SAR-DSM Augsburg/data_SAR_HR.mat",
                        total_filepath="D:/Download/Datasets/Datasets/HS-SAR-DSM Augsburg/TotalImage_transformed.mat"):
    """
    Load the Augsburg dataset using TotalImage for labels.
    
    Returns:
        hs_data: Hyperspectral data
        sar_data: SAR data
        dsm_data: DSM data
        labels: Ground truth labels from TotalImage
        valid_mask: Mask for all valid points (where label > 0)
    """
    print(f"\nLoading Augsburg data...")
    
    # Load DSM data (Digital Surface Model)
    print(f"Loading DSM data from {dsm_filepath}...")
    try:
        data_DSM = read_mat(dsm_filepath)
        dsm = data_DSM["data_DSM"]  # Shape (332, 485)
        print(f"DSM data shape: {dsm.shape}")
    except Exception as e:
        print(f"Error loading DSM data: {e}")
        print("Using dummy DSM data")
        dsm = np.random.rand(332, 485)
    
    # Load Hyperspectral data
    print(f"Loading HS data from {hs_filepath}...")
    try:
        data_HS_LR = read_mat(hs_filepath)
        hs = data_HS_LR["data_HS_LR"]  # Shape (332, 485, 180)
        print(f"HS data shape: {hs.shape}")
    except Exception as e:
        print(f"Error loading HS data: {e}")
        print("Using dummy HS data")
        hs = np.random.rand(332, 485, 180)
    
    # Load SAR data
    print(f"Loading SAR data from {sar_filepath}...")
    try:
        data_SAR_HR = read_mat(sar_filepath)
        sar = data_SAR_HR["data_SAR_HR"]  # Shape (332, 485, 4)
        print(f"SAR data shape: {sar.shape}")
    except Exception as e:
        print(f"Error loading SAR data: {e}")
        print("Using dummy SAR data")
        sar = np.random.rand(332, 485, 4)
    
    # Load TotalImage for ground truth labels
    print(f"Loading label data from {total_filepath}...")
    try:
        total_data = read_mat(total_filepath)
        
        # Find the main data key
        total_key = None
        for key in total_data.keys():
            if not key.startswith('__'):  # Skip metadata
                total_key = key
                break
                
        if total_key:
            labels = total_data[total_key]
            print(f"Labels shape: {labels.shape}")
            print(f"Using key '{total_key}' from TotalImage_transformed.mat")
        else:
            raise KeyError("No valid data key found in TotalImage_transformed.mat")
    except Exception as e:
        print(f"Error loading TotalImage data: {e}")
        print("Using dummy label data")
        labels = np.zeros((332, 485), dtype=int)
        # Randomly select some points for labels
        total_pixels = 332 * 485
        labeled_indices = np.random.choice(total_pixels, size=78000, replace=False)
        labeled_rows, labeled_cols = np.unravel_index(labeled_indices, (332, 485))
        labels[labeled_rows, labeled_cols] = np.random.randint(1, 8, size=78000)
    
    # Create mask for valid points (chỉ nhãn từ 0 đến 6 là valid, nhãn -1 là invalid)
    valid_mask = (labels >= 0)
    
    # Count valid points
    valid_points = np.sum(valid_mask)
    total_pixels = labels.size
    print(f"Total pixels: {total_pixels}")
    print(f"Valid labeled points (label > 0): {valid_points} ({valid_points/total_pixels*100:.1f}%)")
    
    # Print statistics about classes
    unique_classes = np.unique(labels[valid_mask])
    print(f"\nUnique classes in labels: {unique_classes}")
    print(f"Number of classes: {len(unique_classes)}")
    
    # Print distribution of classes
    print("\nClass distribution:")
    for cls in unique_classes:
        count = np.sum(labels == cls)
        print(f"  Class {cls}: {count} pixels ({count/valid_points*100:.2f}%)")
    
    return hs, sar, dsm, labels, valid_mask


def select_supervised_points_per_part(label_parts, valid_mask_parts, supervised_ratio=0.05, min_points_per_part=5):
    """
    Chọn điểm supervised cho mỗi phần ảnh sau khi đã chia.
    
    Args:
        label_parts: List các phần chứa nhãn (TotalImage)
        valid_mask_parts: List các phần mask chỉ ra những điểm có nhãn hợp lệ (label > 0)
        supervised_ratio: Tỷ lệ điểm dùng cho supervised learning
        min_points_per_part: Số điểm supervised tối thiểu cho mỗi phần
        
    Returns:
        training_mask_parts: List các phần mask cho supervised learning
        clustering_mask_parts: List các phần mask cho clustering
    """
    print("\n=== SELECTING SUPERVISED POINTS FOR EACH PART ===")
    
    num_parts = len(label_parts)
    training_mask_parts = []
    clustering_mask_parts = []
    
    total_supervised_points = 0
    total_clustering_points = 0
    
    for part_idx in range(num_parts):
        # Truy cập vào dữ liệu thực trong dictionary
        part_labels = label_parts[part_idx]['data']
        part_valid_mask = valid_mask_parts[part_idx]['data']
        part_name = label_parts[part_idx]['name']
        
        print(f"\nProcessing {part_name}")
        valid_points = np.sum(part_valid_mask)
        
        if valid_points == 0:
            print(f"{part_name}: No valid points")
            # Create empty masks for parts with no valid points
            part_result = {
                'name': part_name,
                'data': np.zeros_like(part_valid_mask),
                'indices': label_parts[part_idx]['indices']
            }
            training_mask_parts.append(part_result)
            clustering_mask_parts.append(part_result.copy())
            continue
        
        # Determine unique classes in this part
        unique_classes = np.unique(part_labels[part_valid_mask])
        
        # Calculate number of supervised points for this part
        # Đảm bảo ít nhất min_points_per_part điểm supervised
        num_supervised = max(min_points_per_part, int(valid_points * supervised_ratio))
        
        # Create masks for this part
        part_training_mask = np.zeros_like(part_valid_mask)
        
        # Find valid point indices in this part
        valid_indices = np.where(part_valid_mask.flatten())[0]
        
        if len(valid_indices) > 0:
            # Calculate how many points to select per class (stratified sampling)
            if len(unique_classes) > 0:
                # Ensure we get at least one point from each class
                points_per_class = max(1, num_supervised // len(unique_classes))
                remaining_points = num_supervised - (points_per_class * len(unique_classes))
                
                # First, select points from each class
                selected_indices = []
                
                for class_id in unique_classes:
                    class_indices = np.where((part_labels.flatten() == class_id) & part_valid_mask.flatten())[0]
                    if len(class_indices) > 0:
                        # Select min(points_per_class, all points in this class)
                        class_selected = np.random.choice(
                            class_indices, 
                            size=min(points_per_class, len(class_indices)), 
                            replace=False
                        )
                        selected_indices.extend(class_selected)
                
                # If we need more points, select randomly from remaining valid points
                if remaining_points > 0 and len(valid_indices) > len(selected_indices):
                    remaining_valid = np.setdiff1d(valid_indices, selected_indices)
                    if len(remaining_valid) > 0:
                        additional = np.random.choice(
                            remaining_valid, 
                            size=min(remaining_points, len(remaining_valid)),
                            replace=False
                        )
                        selected_indices.extend(additional)
            else:
                # If no labeled classes, just select random points
                selected_indices = np.random.choice(
                    valid_indices, 
                    size=min(num_supervised, len(valid_indices)),
                    replace=False
                )
            
            # Set training mask
            rows, cols = np.unravel_index(selected_indices, part_valid_mask.shape)
            part_training_mask[rows, cols] = True
            
            # Check if we got points from all classes
            classes_in_training = np.unique(part_labels[part_training_mask])
            missing_classes = set(unique_classes) - set(classes_in_training)
            
            # If missing classes, add at least one point from each
            if len(missing_classes) > 0:
                for class_id in missing_classes:
                    class_indices = np.where((part_labels == class_id) & part_valid_mask)[0]
                    if len(class_indices) > 0:
                        # Pick a random point
                        idx = np.random.randint(0, len(class_indices))
                        r, c = np.unravel_index(class_indices[idx], part_valid_mask.shape)
                        part_training_mask[r, c] = True
            
            # Set clustering mask (all valid points - only points with label > 0)
            # Chỉ những điểm có nhãn (>0) mới tham gia vào clustering
            # Điểm nhãn 0 bị loại bỏ hoàn toàn, coi như invalid
            part_clustering_mask = part_valid_mask.copy()  # valid_mask đã loại bỏ điểm nhãn 0
            
            # Count supervised points in this part
            supervised_count = np.sum(part_training_mask)
            clustering_count = np.sum(part_clustering_mask)
            total_supervised_points += supervised_count
            total_clustering_points += clustering_count
            
            # Print statistics for this part
            print(f"{part_name}: {supervised_count}/{valid_points} supervised points ({supervised_count/valid_points*100:.2f}%)")
            print(f"  - Target ratio: {supervised_ratio*100:.1f}% per class")
            print(f"  - All {clustering_count} valid points will be used for clustering")
            print(f"  - Classes in this part: {unique_classes}")
            print(f"  - Classes in training set: {np.unique(part_labels[part_training_mask])}")
            
            # Create result dictionaries
            training_result = {
                'name': part_name,
                'data': part_training_mask,
                'indices': label_parts[part_idx]['indices']
            }
            
            clustering_result = {
                'name': part_name,
                'data': part_clustering_mask,
                'indices': label_parts[part_idx]['indices']
            }
            
            # Add masks to result lists
            training_mask_parts.append(training_result)
            clustering_mask_parts.append(clustering_result)
        else:
            # No valid points in this part
            part_result = {
                'name': part_name,
                'data': np.zeros_like(part_valid_mask),
                'indices': label_parts[part_idx]['indices']
            }
            training_mask_parts.append(part_result)
            clustering_mask_parts.append(part_result.copy())
    
    print(f"\nTotal supervised points across all parts: {total_supervised_points}")
    print(f"Total clustering points across all parts: {total_clustering_points}")
    
    return training_mask_parts, clustering_mask_parts


def analyze_classes_per_part(label_parts, training_mask_parts):
    """
    Analyze each image part to determine the number of unique classes in the training data.
    This will be used to set the appropriate number of clusters for each part.
    
    Args:
        label_parts: List of dictionaries containing label data for each part
        training_mask_parts: List of dictionaries containing training masks for each part
        
    Returns:
        Dictionary mapping part names to information about their unique classes
    """
    print("\nAnalyzing classes in each part to determine the number of clusters...")
    clusters_per_part = {}
    
    for idx, label_part_dict in enumerate(label_parts):
        part_name = label_part_dict['name']
        label_part = label_part_dict['data']
        training_mask_part = training_mask_parts[idx]['data'] if training_mask_parts is not None else None
        
        if label_part is not None:
            # Get unique labels from all valid points (>= 0), exclude -1
            unique_classes = np.unique(label_part[label_part >= 0])
            
            # Calculate how many clusters we need
            n_unique_classes = len(unique_classes)
            
            # If no unique classes found, use default
            if n_unique_classes == 0:
                recommended_c = 7  # Default value
                print(f"{part_name}: No valid classes found in training data, using default c = {recommended_c}")
            else:
                recommended_c = n_unique_classes
                print(f"{part_name}: Found {n_unique_classes} unique classes {unique_classes}, setting c = {recommended_c}")
                print(f"  Note: Classes include 0-{n_unique_classes-1} (from original labels 0-{n_unique_classes-1})")
            
            clusters_per_part[part_name] = {
                'unique_classes': unique_classes.tolist() if n_unique_classes > 0 else [],
                'n_unique_classes': n_unique_classes,
                'recommended_c': recommended_c
            }
        else:
            # Default if no label data or training mask
            clusters_per_part[part_name] = {
                'unique_classes': [],
                'n_unique_classes': 0,
                'recommended_c': 7  # Default value
            }
    
    return clusters_per_part


def divide_into_parts(image, labels=None, n_rows=4, n_cols=5):
    """Divide image into n_rows x n_cols parts"""
    height, width = image.shape[:2]
    parts = []
    label_parts = []
    
    # Calculate part dimensions
    part_height = height // n_rows
    part_width = width // n_cols
    
    print(f"Dividing {height}x{width} image into {n_rows}x{n_cols} parts")
    print(f"Each part size is approximately {part_height}x{part_width}")
    
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

def prepare_data_for_processing(hs_part, sar_part, dsm_part, labels_part=None, training_mask_part=None, clustering_mask_part=None, valid_points_mask_part=None, start_h=None, end_h=None, start_w=None, end_w=None):
    """Prepare the data for processing algorithms with Augsburg-specific handling"""
    height, width = hs_part.shape[:2]
    
    # Reshape to 2D for processing (pixels x features)
    hs_reshaped = hs_part.reshape(-1, hs_part.shape[-1])  # (height*width, bands)
    sar_reshaped = sar_part.reshape(-1, sar_part.shape[-1])  # (height*width, bands)
    dsm_reshaped = dsm_part.reshape(-1, 1)  # (height*width, 1)
    
    # Create position matrix for each pixel
    pixel_positions = np.zeros((height, width, 2), dtype=int)
    for i in range(height):
        for j in range(width):
            pixel_positions[i, j] = [i, j]
    pixel_positions_flat = pixel_positions.reshape(-1, 2)
    
    # Filter valid points if we have labels (giống hệt muulf_segmentation_divide.py)
        if labels_part is not None:
            labels_flat = labels_part.flatten()
        valid_indices = labels_flat != -1  # Giống muulf: lọc điểm KHÁC -1
            
            # Count and report
        total_points = len(labels_flat)
        valid_points = np.sum(valid_indices)
        filtered_points = total_points - valid_points
        
        print(f"Filtering unlabeled points...")
            print(f"Total points: {total_points}")
        print(f"Valid labeled points: {valid_points} ({valid_points/total_points*100:.1f}%)")
        print(f"Filtered unlabeled points: {filtered_points} ({filtered_points/total_points*100:.1f}%)")
            
            # Filter data
            hs_reshaped = hs_reshaped[valid_indices]
            sar_reshaped = sar_reshaped[valid_indices]
            dsm_reshaped = dsm_reshaped[valid_indices]
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
    scaler_hs = StandardScaler()
    scaler_sar = StandardScaler()
    scaler_dsm = StandardScaler()
    
    hs_normalized = scaler_hs.fit_transform(hs_reshaped)
    sar_normalized = scaler_sar.fit_transform(sar_reshaped)
    dsm_normalized = scaler_dsm.fit_transform(dsm_reshaped)
    
    # Format data for algorithms (3 views: HS, SAR, DSM)
    data_views = [hs_normalized, sar_normalized, dsm_normalized]
    
    return data_views, (height, width), mask_2d, labels_valid, pixel_positions_valid

def process_part(data_views, labels, c, n_anchors, params):
    """Process one image part"""
    L = len(data_views)  # Number of views
    N = len(data_views[0])  # Number of samples
    
    print(f"\nProcessing image part with {N} samples and {L} views")
    
    # Normalize multiview data
    data = normalize_multiview_data3(data_views)
    
    print("Data after normalization:")
    for i, view in enumerate(data):
        print(f"View {i} shape: {view.shape}")
    
    # Sử dụng MV_DAGL để tạo biểu diễn chung và riêng
    print("Running MV-DAGL...")
    start_time = time.time()
    Z_common, Z_specific = MV_DAGL(data, n_anchors, max_iter=100, alpha=0.2, tol=1e-6)
    end_time = time.time()
    print(f"MV-DAGL completed in {end_time - start_time:.2f} seconds")
    
    print(f"Z_common shape: {Z_common.shape}")
    for i, z in enumerate(Z_specific):
        print(f"Z_specific[{i}] shape: {z.shape}")
    
    # Calculate similarity matrices and prepare data for MPFCM (giống muulf_segmentation_divide.py)
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
    
    # Kiểm tra dữ liệu đầu vào
    print(f"Number of views (L+1): {len(data_new)}")
    print(f"Data shapes: {[d.shape for d in data_new]}")
    print(f"Number of similarity matrices: {len(similarity)}")
    
    # QUAN TRỌNG: Xử lý nhãn đúng cách cho semi-supervised learning
    # Phân tích nhãn trước khi truyền vào MPFCM
    if labels is not None:
        # CHÚ Ý: labels đã được lọc trong prepare_data_for_processing, 
        # nên tất cả các điểm ở đây đều có nhãn >= 0 (valid)
        print(f"\nNhãn sau khi lọc từ prepare_data_for_processing:")
        print(f"- Tổng số điểm: {N}")
        print(f"- Tất cả {N} điểm đều có nhãn valid (>= 0)")
        print(f"- Sẽ sử dụng 5% mỗi nhãn cho supervised learning")
        
        unique_labels = np.unique(labels)
            print(f"- Số lượng lớp duy nhất: {len(unique_labels)}")
            print(f"- Các lớp có trong dữ liệu: {unique_labels}")
            
            # Đếm số điểm trong mỗi lớp
            for label_val in unique_labels:
                count = np.sum(labels == label_val)
            print(f"  + Lớp {label_val}: {count} điểm ({count/N*100:.2f}%)")
    
    # Run MPFCM
    print(f"\nRunning MPFCM with semi-supervised learning...")
    start_time = time.time()
    
    # QUAN TRỌNG: Truyền đúng số view (L) cho run_MPFCM, không phải L+1
    # Vì trong run_MPFCM, L là số view KHÔNG tính common view
    # Chúng ta tự xử lý nhãn cho semi-supervised learning, không dùng hàm create_semi_supervised_labels
    
    # CHÚ Ý: labels đã được lọc trong prepare_data_for_processing
    # Tất cả điểm ở đây đều có nhãn >= 0 (valid)
    if labels is not None:
        # Sử dụng trực tiếp labels đã được lọc
        labels_transformed = labels.copy()
        print("\nSử dụng nhãn đã được lọc (chỉ các điểm valid):")
        unique_labels, counts = np.unique(labels_transformed, return_counts=True)
        print(f"- Phân bố nhãn: {list(zip(unique_labels, counts))}")
        print("- Tất cả nhãn đều valid (>= 0)")
        print("- Sẽ sử dụng 5% mỗi nhãn cho supervised learning")
    else:
        labels_transformed = None
    
    # Sử dụng semi_supervised_ratio=0.05 để tạo supervised/unsupervised split
    # Giống như trong muulf_segmentation_divide.py
    runtime, cluster_predict, cluster_predict_adj = run_MPFCM(L, c, N, data_new, params, similarity, labels_transformed, semi_supervised_ratio=0.05)
    
    # Lưu ý: run_MPFCM đã trả về nhãn cụm ở hệ 1-based (1..c), không cộng thêm
    print("\nGiữ nguyên nhãn cụm 1-based từ run_MPFCM (không cộng thêm 1)")
    
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
            
            # ƯU TIÊN: Tính metrics cho adjusted clustering (u*(1-xi))
            from sklearn.metrics import cohen_kappa_score
            
            # Metrics for adjusted clustering (u*(1-xi))
            valid_clusters_adj = cluster_predict_adj[valid_indices]
            # Lưu ý: valid_labels và valid_clusters_adj đều trong khoảng 1-7
            ari_adj = adjusted_rand_score(valid_labels, valid_clusters_adj)
            metrics_adj["ari"] = ari_adj
            nmi_adj = normalized_mutual_info_score(valid_labels, valid_clusters_adj)
            metrics_adj["nmi"] = nmi_adj
            # Thêm kappa coefficient
            kappa_adj = cohen_kappa_score(valid_labels, valid_clusters_adj)
            metrics_adj["kappa"] = kappa_adj
            
            # In thông tin về số lượng điểm đánh giá
            total_valid_points = np.sum(valid_indices)
            metrics_adj["valid_points"] = int(total_valid_points)
            print(f"\nTổng số điểm có ground truth (nhãn > 0) được đánh giá: {total_valid_points}")
            
            print("\nSupervised metrics for adjusted clustering (from u*(1-xi)):")
            print(f"  - Adjusted Rand Index: {ari_adj:.4f} (higher is better)")
            print(f"  - Normalized Mutual Information: {nmi_adj:.4f} (higher is better)")
            print(f"  - Kappa Coefficient: {kappa_adj:.4f} (higher is better)")
            
            # Metrics for standard clustering (u)
            valid_clusters = cluster_predict[valid_indices]
            # Lưu ý: valid_labels và valid_clusters đều trong khoảng 1-7
            ari = adjusted_rand_score(valid_labels, valid_clusters)
            metrics["ari"] = ari
            nmi = normalized_mutual_info_score(valid_labels, valid_clusters)
            metrics["nmi"] = nmi
            # Thêm kappa coefficient
            kappa = cohen_kappa_score(valid_labels, valid_clusters)
            metrics["kappa"] = kappa
            
            # Lưu thông tin về số lượng điểm đánh giá
            metrics["valid_points"] = int(total_valid_points)
            
            print("\nSupervised metrics for standard clustering (from u):")
            print(f"  - Adjusted Rand Index: {ari:.4f} (higher is better)")
            print(f"  - Normalized Mutual Information: {nmi:.4f} (higher is better)")
            print(f"  - Kappa Coefficient: {kappa:.4f} (higher is better)")
        except Exception as e:
            print(f"Error calculating supervised metrics: {e}")
    
    # Thay đổi thứ tự trả về, ưu tiên cluster_predict_adj và metrics_adj
    return cluster_predict_adj, cluster_predict, metrics_adj, metrics

def main(force_reload=True, supervised_ratio=0.05, min_points_per_part=5):
    print("Starting Augsburg Image Segmentation using MPFCM with MV_DAGL using TotalImage...")
    
    # Load Augsburg data - now using TotalImage for labels
    hs_data, sar_data, dsm_data, labels, valid_mask = load_augsburg_data()
    
    if hs_data is None or sar_data is None or dsm_data is None:
        print("Error loading data. Exiting.")
        return
    
    # Get original dimensions
    height, width = hs_data.shape[:2]
    print(f"Original image dimensions: {height}x{width}")
    
    # Define the number of parts in each dimension
    n_rows = 2
    n_cols = 2
    
    # Calculate expected part size
    part_height = height // n_rows
    part_width = width // n_cols
    print(f"Expected part dimensions: {part_height}x{part_width} pixels ({part_height*part_width} total pixels per part)")
    
    # Divide image into parts
    print(f"Dividing image into {n_rows}x{n_cols} parts...")
    hs_parts, label_parts = divide_into_parts(hs_data, labels, n_rows, n_cols)
    sar_parts, _ = divide_into_parts(sar_data, None, n_rows, n_cols)
    dsm_parts, _ = divide_into_parts(dsm_data, None, n_rows, n_cols)
    
    # Tạo valid mask cho từng part dựa trên labels trong part đó
    valid_mask_parts = []
    for idx, label_part_dict in enumerate(label_parts):
        label_part = label_part_dict['data']
        # Tạo valid mask cho part này: điểm có nhãn >= 0
        part_valid_mask = (label_part >= 0)
        
        valid_mask_parts.append({
            'data': part_valid_mask,
            'position': label_part_dict['position'],
            'indices': label_part_dict['indices'],
            'name': label_part_dict['name']
        })
        
        # Debug info
        total_points_in_part = label_part.size
        valid_points_in_part = np.sum(part_valid_mask)
        print(f"Part {label_part_dict['name']}: {valid_points_in_part}/{total_points_in_part} valid points")
        
        if valid_points_in_part > 0:
            valid_labels_in_part = label_part[part_valid_mask]
            unique_labels_in_part = np.unique(valid_labels_in_part)
            print(f"  Valid labels: {unique_labels_in_part}")
    
    # Bỏ qua việc tạo training_mask vì sẽ sử dụng semi_supervised_ratio trong run_MPFCM
    # training_mask_parts, clustering_mask_parts = select_supervised_points_per_part(
    #     label_parts, valid_mask_parts, supervised_ratio, min_points_per_part
    # )
    
    # Phân tích số lượng classes trong mỗi phần và xác định số clusters phù hợp
    # Không cần training_mask vì sẽ sử dụng semi_supervised_ratio
    clusters_per_part = analyze_classes_per_part(label_parts, None)
    
    # Analyze ground truth clusters in each part BEFORE clustering
    print("\n\n" + "="*50)
    print("ANALYZING GROUND TRUTH CLUSTERS IN EACH PART (BEFORE CLUSTERING)")
    print("="*50)
    
    # Create a table header for ground truth analysis
    print(f"{'Part Name':<15} | {'Total Pixels':<12} | {'Valid Points':<12} | {'Unique Clusters':<15} | {'Cluster Distribution':<50}")
    print("-" * 105)
    
    # Analyze each part for ground truth clusters
    for idx, part in enumerate(hs_parts):
        part_name = part['name']
        part_shape = part['data'].shape
        total_pixels = part_shape[0] * part_shape[1]
        
        # Get valid points mask and labels for this part
        valid_points_mask_part = valid_mask_parts[idx]['data'] if valid_mask_parts is not None else None
        label_part = label_parts[idx]['data'] if label_parts is not None else None
        
        if valid_points_mask_part is not None and label_part is not None:
            # Get only the valid points and their labels
            valid_points = np.sum(valid_points_mask_part)
            
            # Extract labels of valid points
            valid_labels = label_part[valid_points_mask_part]
            
            # Find unique clusters (excluding negative values)
            unique_clusters = np.unique(valid_labels[valid_labels >= 0])
                n_unique_clusters = len(unique_clusters)
                
                # Calculate cluster distribution
                cluster_distribution = {}
                for cluster in unique_clusters:
                count = np.sum(valid_labels == cluster)
                    cluster_distribution[int(cluster)] = count
                
                # Format cluster distribution as string
                if cluster_distribution:
                    cluster_dist_str = ", ".join([f"{c}:{n}" for c, n in sorted(cluster_distribution.items())])
                    if len(cluster_dist_str) > 50:
                        cluster_dist_str = cluster_dist_str[:47] + "..."
                else:
                cluster_dist_str = "No valid labels"
            
            # Print row
            print(f"{part_name:<15} | {total_pixels:<12} | {valid_points:<12} | {n_unique_clusters:<15} | {cluster_dist_str:<50}")
        else:
            print(f"{part_name:<15} | {total_pixels:<12} | {'N/A':<12} | {'N/A':<15} | {'N/A':<50}")
    
    print("="*105)
    
    # Save detailed cluster information for each part
    print("\nSaving detailed ground truth cluster information for each part...")
    os.makedirs('parts_augsburg_ground_truth', exist_ok=True)
    
    # [phần code phân tích ground truth vẫn giữ nguyên]
    
    # Continue with the regular analysis of points distribution
    print("\n\n" + "="*50)
    print("ANALYZING DISTRIBUTION OF POINTS ACROSS ALL PARTS")
    print("="*50)
    
    # Create a table header
    print(f"{'Part Name':<15} | {'Total Pixels':<12} | {'Valid Pts':<12} | {'Valid %':<8}")
    print("-" * 50)
    
    # Loop through all parts
    for idx, part in enumerate(hs_parts):
        part_name = part['name']
        part_shape = part['data'].shape
        total_pixels = part_shape[0] * part_shape[1]
        
        # Get counts for this part
        valid_pts = np.sum(valid_mask_parts[idx]['data']) if valid_mask_parts is not None else 0
        valid_pct = (valid_pts / total_pixels) * 100 if total_pixels > 0 else 0
        
        # Print row
        print(f"{part_name:<15} | {total_pixels:<12} | {valid_pts:<12} | {valid_pct:<8.2f}%")
    
    print("="*50)
    
    # Print total statistics
    # [phần code thống kê vẫn giữ nguyên]
    
    # Create output directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('parts_augsburg', exist_ok=True)
    os.makedirs('parts_augsburg_clusters', exist_ok=True)
    
    # Process each part
    parts_results = []
    
    for idx, (hs_part, sar_part, dsm_part) in enumerate(zip(hs_parts, sar_parts, dsm_parts)):
        part_name = hs_part['name']
        print(f"\n\n{'='*50}")
        print(f"Processing part {idx+1}/{len(hs_parts)}: {part_name}")
        print(f"Position: {hs_part['position']}")
        print(f"Indices: {hs_part['indices']}")
        print(f"Shape: {hs_part['data'].shape}")
        print(f"Total pixels: {hs_part['data'].shape[0] * hs_part['data'].shape[1]}")
        
        # Get corresponding label part and valid points mask
        label_part = label_parts[idx]['data'] if label_parts is not None else None
        valid_points_mask_part = valid_mask_parts[idx]['data'] if valid_mask_parts is not None else None
        
        # Count valid points in this part
        if valid_points_mask_part is not None:
            valid_points = np.sum(valid_points_mask_part)
            print(f"Valid points (label >= 0) in this part: {valid_points}")
            print(f"Note: 5% of these points will be used for supervised learning")
        
        # Sử dụng số clusters đã phân tích hoặc giá trị mặc định
        if part_name in clusters_per_part:
            c = clusters_per_part[part_name]['recommended_c']
            print(f"Using recommended number of clusters: c = {c} based on unique classes: {clusters_per_part[part_name]['unique_classes']}")
        else:
            c = 7  # Giá trị mặc định
            print(f"Using default number of clusters: c = {c}")
        
        # Parameters
        n_anchors = 4*c  # Number of anchors
        
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
        
        # Check if we can load pre-existing data for this part
        existing_hs, existing_sar, existing_dsm, existing_labels, existing_training_mask, existing_clustering_mask, existing_valid_points_mask = load_part_data(part_name)
        
        # Debug message to help identify the issue
        print(f"DEBUG: Checking existing data for {part_name}")
        if existing_valid_points_mask is not None:
            print(f"DEBUG: existing_valid_points_mask sum: {np.sum(existing_valid_points_mask)}")
        if existing_clustering_mask is not None:
            print(f"DEBUG: existing_clustering_mask sum: {np.sum(existing_clustering_mask)}")
        if existing_training_mask is not None:
            print(f"DEBUG: existing_training_mask sum: {np.sum(existing_training_mask)}")
            
        if force_reload or existing_hs is None:
            # Use the current part data
            current_hs = hs_part['data']
            current_sar = sar_part['data']
            current_dsm = dsm_part['data']
            current_labels = label_part
            current_valid_points_mask = valid_points_mask_part
            
            # Save part data for future use
            save_part_data(
                part_name,
                current_hs,
                current_sar,
                current_dsm,
                current_labels,
                None,  # No training_mask
                None,  # No clustering_mask
                current_valid_points_mask,
                hs_part['indices']
            )
        else:
            print(f"Using existing data for {part_name} from disk...")
            current_hs = existing_hs
            current_sar = existing_sar
            current_dsm = existing_dsm
            current_labels = existing_labels
            current_valid_points_mask = existing_valid_points_mask
        
        # Prepare data for processing (extract features and normalize) - giống muulf_segmentation_divide.py
        data_views, part_shape, mask_2d, labels_valid, pixel_positions = prepare_data_for_processing(
            current_hs,
            current_sar,
            current_dsm,
            current_labels
        )
        
        sample_size = len(data_views[0]) if data_views else 0
        print(f"Number of samples after preparation: {sample_size}")
        
        if sample_size == 0:
            print(f"No valid samples in {part_name}, skipping...")
            continue
        
        # Kiểm tra kích thước của mỗi view
        for i, view in enumerate(data_views):
            print(f"View {i} shape: {view.shape}")
            
        # Process this part - Lưu ý thứ tự trả về đã thay đổi để ưu tiên kết quả adjusted
        cluster_predict_adj, cluster_predict, metrics_adj, metrics = process_part(
            data_views, labels_valid, c, n_anchors, params
        )
        
        # Visualize cluster assignments for this part - ƯU TIÊN adjusted clustering
        part_height, part_width = current_hs.shape[:2]
        
        # Hiển thị adjusted cluster assignments trước (u*(1-xi))
        adj_cluster_image = None
        if cluster_predict_adj is not None and len(cluster_predict_adj) > 0:
            adj_cluster_image = visualize_part_clustering(
                f"{part_name}_adjusted", 
                (part_height, part_width), 
                cluster_predict_adj, 
                pixel_positions,
                save_dir='parts_augsburg_clusters_adjusted'
            )
        
        # Sau đó hiển thị standard clustering (u)
        cluster_image = visualize_part_clustering(
            part_name, 
            (part_height, part_width), 
            cluster_predict, 
            pixel_positions
            )
        
        # Store results
        parts_results.append({
            'part_idx': idx,
            'position': hs_part['position'],
            'indices': hs_part['indices'],
            'name': part_name,
            'cluster_predict': cluster_predict,
            'cluster_predict_adj': cluster_predict_adj,
            'metrics': metrics,
            'metrics_adj': metrics_adj,
            'pixel_positions': pixel_positions,
            'cluster_image': cluster_image
        })
    
    # Visualize and save the combined results
    full_cluster_image = visualize_part_results(
        (height, width), hs_data, sar_data, dsm_data, parts_results, n_rows, n_cols
    )
    
    print(f"\nAugsburg {n_rows}x{n_cols} Parts segmentation completed!")
    print(f"Results saved to:")
    print(f"- results/augsburg_mpfcm_parts_{n_rows}x{n_cols}.png")
    print(f"- results/augsburg_mpfcm_labels_parts_{n_rows}x{n_cols}.npy")
    print(f"- Individual part data saved in parts_augsburg/ directory")
    print(f"- Individual cluster results saved in parts_augsburg_clusters/ directory")
    
    # Create a verification image showing parts divisions
    verification = np.zeros((height, width, 3))
    
    # Use SAR data for visualization
    sar_vis = sar_data[:, :, :3] if sar_data.shape[2] >= 3 else np.zeros((height, width, 3))
    # Normalize
    sar_vis = (sar_vis - np.min(sar_vis)) / (np.max(sar_vis) - np.min(sar_vis))
    verification = sar_vis.copy()
    
    # Draw horizontal borders
    for i in range(1, n_rows):
        border_y = i * (height // n_rows)
        verification[border_y-1:border_y+1, :] = [1, 0, 0]  # Red horizontal lines
    
    # Draw vertical borders
    for j in range(1, n_cols):
        border_x = j * (width // n_cols)
        verification[:, border_x-1:border_x+1] = [1, 0, 0]  # Red vertical lines
    
    # Save verification image
    plt.figure(figsize=(10, 10))
    plt.imshow(verification)
    plt.title(f'Augsburg Image Division into {n_rows}x{n_cols} Parts')
    plt.axis('off')
    plt.savefig('parts_augsburg/verification.png', dpi=300, bbox_inches='tight')
    
    # Đánh giá kết quả phân cụm tổng hợp
    metrics_adj, metrics = evaluate_overall_clustering(hs_data, labels, parts_results, n_rows, n_cols, valid_mask)
    
    # In thông tin tóm tắt cuối cùng
    print("\n" + "="*80)
    print("KẾT QUẢ CUỐI CÙNG:")
    print("="*80)
    # ƯU TIÊN hiển thị kết quả cho u*(1-xi)
    if metrics_adj is not None:
        print("Kết quả phân cụm từ u*(1-xi):")
        print(f"Accuracy: {metrics_adj['accuracy']:.4f}")
        print(f"F1-score: {metrics_adj['f1_score']:.4f}")
        print(f"ARI: {metrics_adj['adjusted_rand_index']:.4f}")
        if 'kappa' in metrics_adj:
            print(f"Kappa: {metrics_adj['kappa']:.4f}")
    
    # Hiển thị kết quả cho u để so sánh
    if metrics is not None:
        print("\nKết quả phân cụm từ u (để so sánh):")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-score: {metrics['f1_score']:.4f}")
        print(f"ARI: {metrics['adjusted_rand_index']:.4f}")
        if 'kappa' in metrics:
            print(f"Kappa: {metrics['kappa']:.4f}")
    # Tạo ảnh ghép lại từ 4 phần theo u*(1-xi)
    print("\n" + "="*60)
    print("CREATING COMBINED IMAGE FROM 4 PARTS (u*(1-xi) RESULTS)")
    print("="*60)
    
    try:
        combined_image = create_combined_image_from_parts()
        
        # Thống kê kết quả
        print("\n" + "="*60)
        print("COMBINED IMAGE STATISTICS:")
        print("="*60)
        
        total_pixels = combined_image.size
        valid_pixels = np.sum(combined_image >= 0)
        invalid_pixels = total_pixels - valid_pixels
        
        print(f"Total pixels: {total_pixels}")
        print(f"Valid pixels (clustered): {valid_pixels} ({valid_pixels/total_pixels*100:.2f}%)")
        print(f"Invalid pixels (no cluster): {invalid_pixels} ({invalid_pixels/total_pixels*100:.2f}%)")
        
        if valid_pixels > 0:
            unique_clusters = np.unique(combined_image[combined_image >= 0])
            print(f"Number of clusters: {len(unique_clusters)}")
            print(f"Cluster IDs: {unique_clusters}")
            
            print("\nCluster distribution:")
            for cluster_id in unique_clusters:
                count = np.sum(combined_image == cluster_id)
                percentage = count / valid_pixels * 100
                print(f"  Cluster {cluster_id}: {count} pixels ({percentage:.2f}%)")
        
        print("\nFiles generated:")
        print("- results/augsburg_combined_4parts_u1xi.npy (combined cluster assignments)")
        print("- results/augsburg_combined_4parts_u1xi.png (visualization)")
        print("- results/augsburg_combined_4parts_u1xi_overlay.png (overlay on SAR)")
        
    except Exception as e:
        print(f"Error creating combined image: {e}")
    
    print("="*80)

def create_combined_image_from_parts():
    """
    Tạo ảnh ghép lại từ 4 phần (top_left, top_right, bottom_left, bottom_right) 
    sử dụng kết quả u*(1-xi) từ thư mục parts_augsburg_clusters_adjusted
    """
    print("Creating combined image from 4 parts using u*(1-xi) results...")
    
    # Đường dẫn đến các file cluster đã được adjusted
    parts_dir = "parts_augsburg_clusters_adjusted"
    
    # Tên các phần
    part_names = ["top_left", "top_right", "bottom_left", "bottom_right"]
    
    # Kích thước ảnh gốc (332, 485)
    full_height, full_width = 332, 485
    
    # Kích thước mỗi phần (chia 2x2)
    part_height = full_height // 2
    part_width = full_width // 2
    
    print(f"Full image dimensions: {full_height}x{full_width}")
    print(f"Part dimensions: {part_height}x{part_width}")
    
    # Tạo ảnh trống để ghép
    combined_image = np.zeros((full_height, full_width), dtype=int) - 1
    
    # Định nghĩa vị trí của từng phần
    part_positions = {
        "top_left": (0, 0, part_height, part_width),
        "top_right": (0, part_width, part_height, full_width),
        "bottom_left": (part_height, 0, full_height, part_width),
        "bottom_right": (part_height, part_width, full_height, full_width)
    }
    
    # Ghép từng phần vào ảnh tổng hợp
    for part_name in part_names:
        # Thử tìm file adjusted clusters trước
        cluster_file = os.path.join(parts_dir, f"{part_name}_adjusted_clusters.npy")
        
        if not os.path.exists(cluster_file):
            # Nếu không có, thử tìm file clusters thường
            cluster_file = os.path.join(parts_dir, f"{part_name}_clusters.npy")
        
        if os.path.exists(cluster_file):
            print(f"Loading {part_name} from {cluster_file}")
            part_clusters = np.load(cluster_file)
            
            # Lấy vị trí của phần này
            start_h, start_w, end_h, end_w = part_positions[part_name]
            
            # Ghép vào ảnh tổng hợp
            combined_image[start_h:end_h, start_w:end_w] = part_clusters
            
            print(f"  Added {part_name} at position ({start_h}:{end_h}, {start_w}:{end_w})")
            print(f"  Part shape: {part_clusters.shape}")
            print(f"  Unique clusters in part: {np.unique(part_clusters[part_clusters >= 0])}")
        else:
            print(f"Warning: {cluster_file} not found")
    
    # Tạo thư mục results nếu chưa có
    os.makedirs("results", exist_ok=True)
    
    # Lưu ảnh ghép
    np.save("results/augsburg_combined_4parts_u1xi.npy", combined_image)
    print(f"Saved combined image: results/augsburg_combined_4parts_u1xi.npy")
    
    # Tạo visualization
    create_combined_visualization(combined_image, "results/augsburg_combined_4parts_u1xi.png")
    
    return combined_image

def create_combined_visualization(cluster_image, save_path):
    """
    Tạo visualization cho ảnh ghép
    """
    height, width = cluster_image.shape
    
    # Lấy các cluster duy nhất (loại bỏ -1)
    unique_clusters = np.unique(cluster_image[cluster_image >= 0])
    num_clusters = len(unique_clusters)
    
    print(f"Creating visualization with {num_clusters} clusters: {unique_clusters}")
    
    # Tạo colormap
    colors = plt.cm.jet(np.linspace(0, 1, num_clusters))
    
    # Tạo ảnh màu
    colored_image = np.zeros((height, width, 3))
    
    # Map cluster indices to colors
    for i, cluster in enumerate(unique_clusters):
        mask = cluster_image == cluster
        colored_image[mask] = colors[i, :3]
    
    # Lưu ảnh
    plt.figure(figsize=(12, 10))
    plt.imshow(colored_image)
    plt.title('Augsburg Combined Image (4 Parts) - u*(1-xi) Results')
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization: {save_path}")
    
    # Tạo thêm ảnh với overlay trên SAR data
    create_overlay_visualization(cluster_image, save_path.replace('.png', '_overlay.png'))

def create_overlay_visualization(cluster_image, save_path):
    """
    Tạo visualization với overlay trên SAR data
    """
    try:
        # Load SAR data
        sar_filepath = "D:/Download/Datasets/Datasets/HS-SAR-DSM Augsburg/data_SAR_HR.mat"
        sar_data = read_mat(sar_filepath)
        
        if sar_data is not None and "data_SAR_HR" in sar_data:
            sar = sar_data["data_SAR_HR"]
            
            # Sử dụng 3 kênh đầu tiên của SAR
            if sar.shape[2] >= 3:
                sar_vis = sar[:, :, :3].copy()
                # Normalize
                sar_vis = (sar_vis - np.min(sar_vis)) / (np.max(sar_vis) - np.min(sar_vis))
                
                height, width = cluster_image.shape
                
                # Lấy các cluster duy nhất
                unique_clusters = np.unique(cluster_image[cluster_image >= 0])
                num_clusters = len(unique_clusters)
                colors = plt.cm.jet(np.linspace(0, 1, num_clusters))
                
                # Tạo overlay image (70% clustering, 30% SAR)
                overlay_image = np.zeros((height, width, 3))
                
                for i in range(height):
                    for j in range(width):
                        cluster = cluster_image[i, j]
                        if cluster >= 0:
                            color_idx = np.where(unique_clusters == cluster)[0][0]
                            overlay_image[i, j] = 0.7 * colors[color_idx, :3] + 0.3 * sar_vis[i, j]
                        else:
                            overlay_image[i, j] = sar_vis[i, j]
                
                # Lưu overlay image
                plt.figure(figsize=(12, 10))
                plt.imshow(overlay_image)
                plt.title('Augsburg Combined Image (4 Parts) - u*(1-xi) Results Overlaid on SAR')
                plt.axis('off')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Saved overlay visualization: {save_path}")
            else:
                print("SAR data doesn't have enough channels for visualization")
        else:
            print("Could not load SAR data for overlay")
            
    except Exception as e:
        print(f"Error creating overlay visualization: {e}")

if __name__ == "__main__":
    main()