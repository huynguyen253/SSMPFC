import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from pymatreader import read_mat
import cv2
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

def resize_labels(labels, new_size):
    """Resize label matrix using nearest neighbor interpolation
    
    Args:
        labels: Original labels array with shape (height, width)
        new_size: Target size as (height, width) in NumPy format
        
    Returns:
        Resized labels array with shape new_size
    """
    # OpenCV expects (width, height) while our new_size is (height, width)
    cv2_size = (new_size[1], new_size[0])
    
    print(f"Resizing labels from {labels.shape} to {new_size}")
    print(f"Using OpenCV size format: {cv2_size} (width, height)")
    
    resized = cv2.resize(labels.astype(np.float32), cv2_size, interpolation=cv2.INTER_NEAREST)
    return np.round(resized).astype(np.int32)

def load_ground_truth(filepath, new_size=(82, 55)):
    """Load original ground truth labels and resize to match segmentation result"""
    print(f"Loading ground truth from {filepath}...")
    data = read_mat(filepath)
    hsi = data["hsi"]
    
    try:
        labels = hsi["sceneLabels"]["labels"]
        print(f"Original labels shape: {labels.shape}")
        
        # Resize labels to match segmentation result
        resized_labels = resize_labels(labels, new_size)
        print(f"Resized labels shape: {resized_labels.shape}")
        
        # Check label distribution
        unique_labels = np.unique(resized_labels)
        print(f"Unique labels in ground truth: {unique_labels}")
        
        hist = {}
        for label in unique_labels:
            if label != -1:  # Skip unlabeled
                hist[int(label)] = np.sum(resized_labels == label)
        print(f"Label histogram: {hist}")
        
        return resized_labels
    except (KeyError, IndexError) as e:
        print(f"Error loading ground truth: {e}")
        return None

def evaluate_segmentation(segmentation, ground_truth):
    """Evaluate segmentation results against ground truth"""
    # Reshape to 1D arrays
    seg_flat = segmentation.flatten()
    gt_flat = ground_truth.flatten()
    
    # Filter out unlabeled points (-1)
    valid_indices = gt_flat != -1
    valid_gt = gt_flat[valid_indices]
    valid_seg = seg_flat[valid_indices]
    
    print(f"Total points: {len(gt_flat)}")
    print(f"Valid labeled points: {np.sum(valid_indices)} ({np.sum(valid_indices)/len(gt_flat)*100:.1f}%)")
    
    # Get unique classes and clusters
    unique_classes = np.sort(np.unique(valid_gt))  # Sorted to maintain order
    unique_clusters = np.sort(np.unique(valid_seg))  # Sorted to maintain order
    
    print(f"Number of true classes: {len(unique_classes)}")
    print(f"Number of predicted clusters: {len(unique_clusters)}")
    
    # Check if segmentation matches ground truth exactly
    exact_match_ratio = np.sum(valid_seg == valid_gt) / len(valid_gt)
    print(f"Exact match ratio before any mapping: {exact_match_ratio:.4f}")
    
    # Create a proper confusion matrix (rows=classes, columns=clusters)
    conf_mat = np.zeros((len(unique_classes), len(unique_clusters)))
    
    # Populate confusion matrix
    for i, c in enumerate(unique_classes):
        for j, k in enumerate(unique_clusters):
            conf_mat[i, j] = np.sum((valid_gt == c) & (valid_seg == k))
    
    print("Confusion matrix before mapping (sample):")
    print(conf_mat[:5, :5] if conf_mat.shape[0] > 5 and conf_mat.shape[1] > 5 else conf_mat)
    
    # Find optimal assignment using Hungarian algorithm
    # The algorithm minimizes cost, so we negate the confusion matrix (maximize overlap)
    row_ind, col_ind = linear_sum_assignment(-conf_mat)
    
    # Create mapping from cluster ID to true class
    cluster_to_class = {}
    for i in range(len(row_ind)):
        cluster_id = unique_clusters[col_ind[i]]
        class_id = unique_classes[row_ind[i]]
        cluster_to_class[cluster_id] = class_id
    
    print("\nCluster mapping to true classes:")
    for cluster_id, class_id in sorted(cluster_to_class.items()):
        count = np.sum(valid_seg == cluster_id)
        print(f"  Cluster {cluster_id} → Class {class_id} ({count} points)")
    
    # Create mapped predictions
    mapped_seg = np.full_like(segmentation, -1)  # Start with all -1 (unlabeled)
    for cluster_id, class_id in cluster_to_class.items():
        mapped_seg[segmentation == cluster_id] = class_id
    
    # Convert flattened predictions for metrics calculation
    valid_mapped_seg = mapped_seg.flatten()[valid_indices]
    
    # Calculate metrics
    metrics = {}
    
    # Metrics that don't need mapping
    metrics["ari"] = adjusted_rand_score(valid_gt, valid_seg)
    metrics["nmi"] = normalized_mutual_info_score(valid_gt, valid_seg)
    
    # Metrics that need mapping
    accuracy = np.sum(valid_gt == valid_mapped_seg) / len(valid_gt)
    metrics["accuracy"] = accuracy
    
    # F1 score with the mapped predictions
    f1 = f1_score(valid_gt, valid_mapped_seg, average='weighted')
    metrics["f1_score"] = f1
    
    # Print metrics
    print("\nClustering Evaluation Results:")
    print(f"  - Accuracy after optimal mapping: {accuracy:.4f}")
    print(f"  - Adjusted Rand Index (ARI): {metrics['ari']:.4f}")
    print(f"  - Normalized Mutual Information (NMI): {metrics['nmi']:.4f}")
    print(f"  - F1 Score: {metrics['f1_score']:.4f}")
    
    # Calculate confusion matrix with mapped predictions
    conf_mat_mapped = confusion_matrix(valid_gt, valid_mapped_seg)
    
    return metrics, conf_mat_mapped, cluster_to_class

def visualize_results(segmentation, ground_truth, cluster_to_class, save_path=None):
    """Visualize segmentation results compared to ground truth"""
    # Create mapped segmentation for visualization
    mapped_seg = np.copy(segmentation)
    
    # Map each cluster to its corresponding class
    for cluster_id, class_id in cluster_to_class.items():
        mapped_seg[segmentation == cluster_id] = class_id
    
    # Set unlabeled points to -1 in both images
    mapped_seg[ground_truth == -1] = -1
    
    # Create a visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Create a consistent colormap
    cmap = plt.cm.get_cmap('tab20', np.max([np.max(ground_truth), np.max(segmentation), np.max(mapped_seg)])+1)
    cmap.set_under('black')  # Color for -1 (unlabeled)
    
    # Original ground truth
    im0 = axes[0].imshow(ground_truth, cmap=cmap, interpolation='none', vmin=0)
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    
    # Original segmentation (before mapping)
    im1 = axes[1].imshow(segmentation, cmap=cmap, interpolation='none', vmin=0)
    axes[1].set_title('Segmentation Result')
    axes[1].axis('off')
    
    # Mapped segmentation
    im2 = axes[2].imshow(mapped_seg, cmap=cmap, interpolation='none', vmin=0)
    axes[2].set_title('Mapped Segmentation')
    axes[2].axis('off')
    
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def compare_segmentation_with_gt(segmentation_path, gt_path, new_size=(82, 55)):
    """Compare the segmentation result directly with ground truth to check for data issues"""
    # Load segmentation
    print(f"Loading segmentation from {segmentation_path}...")
    segmentation = np.load(segmentation_path)
    print(f"Segmentation shape: {segmentation.shape}")
    
    # Load ground truth
    print(f"Loading ground truth from {gt_path}...")
    data = read_mat(gt_path)
    gt_labels = data["hsi"]["sceneLabels"]["labels"]
    print(f"Original ground truth shape: {gt_labels.shape}")
    
    # Resize ground truth to match segmentation
    cv2_size = (new_size[1], new_size[0])  # OpenCV expects (width, height)
    resized_gt = cv2.resize(gt_labels.astype(np.float32), cv2_size, interpolation=cv2.INTER_NEAREST)
    resized_gt = np.round(resized_gt).astype(np.int32)
    print(f"Resized ground truth shape: {resized_gt.shape}")
    
    # Flatten both arrays
    seg_flat = segmentation.flatten()
    gt_flat = resized_gt.flatten()
    
    # Filter out unlabeled points (-1)
    valid_indices = gt_flat != -1
    valid_gt = gt_flat[valid_indices]
    valid_seg = seg_flat[valid_indices]
    
    # Check for direct equality
    direct_match_count = np.sum(valid_seg == valid_gt)
    direct_match_ratio = direct_match_count / len(valid_gt)
    print(f"Direct match ratio: {direct_match_ratio:.4f} ({direct_match_count}/{len(valid_gt)} points)")
    
    if direct_match_ratio > 0.9:
        print("\nWARNING: The segmentation result is extremely similar to the ground truth!")
        print("This suggests the algorithm might be using ground truth labels during clustering,")
        print("which would make this a supervised rather than unsupervised approach.")
        print("Check if the labels are being used inappropriately in the clustering process.")
    
    # Check for possible label offset
    # Sometimes segmentation might be ground truth with a constant offset
    if direct_match_ratio < 0.9:  # Only check if not already high match
        differences = valid_seg - valid_gt
        unique_diffs, diff_counts = np.unique(differences, return_counts=True)
        
        print(f"Unique differences between segmentation and ground truth:")
        for diff, count in zip(unique_diffs, diff_counts):
            print(f"  Difference {diff}: {count} points ({count/len(valid_gt)*100:.1f}%)")
    
    return direct_match_ratio

def main():
    # Load segmentation result
    seg_path = "results/muufl_mpfcm_labels_full_image_resized.npy"
    gt_path = "data/muufl_gulfport_campus_1_hsi_220_label.mat"
    
    # First check if the segmentation might be derived from ground truth
    direct_match_ratio = compare_segmentation_with_gt(seg_path, gt_path)
    
    print("\n" + "="*50)
    print("Beginning detailed evaluation...")
    print("="*50 + "\n")
    
    print(f"Loading segmentation result from {seg_path}...")
    segmentation = np.load(seg_path)
    print(f"Segmentation shape: {segmentation.shape}")
    
    # Ensure height, width are consistent with our understanding
    height, width = segmentation.shape
    print(f"Height: {height}, Width: {width}")
    
    # Load ground truth and resize to match segmentation
    ground_truth = load_ground_truth(gt_path, new_size=(height, width))
    
    if ground_truth is None:
        print("Could not load ground truth. Exiting.")
        return
    
    # Verify shapes match before proceeding
    if ground_truth.shape != segmentation.shape:
        print(f"Error: Shape mismatch between ground_truth {ground_truth.shape} and segmentation {segmentation.shape}")
        return
        
    # Evaluate segmentation
    metrics, conf_mat, cluster_to_class = evaluate_segmentation(segmentation, ground_truth)
    
    # Visualize results
    visualize_results(segmentation, ground_truth, cluster_to_class, save_path="results/segmentation_evaluation.png")
    
    # Save confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_mat, cmap='Blues', interpolation='none')
    plt.colorbar(label='True Positive Rate')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Normalized Confusion Matrix')
    
    # Add text annotations
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            text_color = 'white' if conf_mat[i, j] > 0.5 else 'black'
            plt.text(j, i, f'{conf_mat[i, j]:.2f}', ha='center', va='center', color=text_color)
    
    plt.tight_layout()
    plt.savefig("results/confusion_matrix_evaluation.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main() 