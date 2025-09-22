import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pymatreader import read_mat


def resize_image_nn(image: np.ndarray, new_size: tuple[int, int]) -> np.ndarray:
    """Resize 2D integer label image using nearest-neighbor (width, height)."""
    return cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)


def load_muufl_labels(filepath: str, new_size):
    """Load MUUFL scene labels from .mat; optionally resize to new_size (w, h).

    Returns labels as int32; unlabeled kept as -1 if present.
    """
    data = read_mat(filepath)
    hsi = data.get("hsi", {})

    labels = None
    try:
        labels = hsi["sceneLabels"]["labels"]
    except Exception:
        # Try alternative common key
        labels = hsi.get("labels", None)

    if labels is None:
        raise RuntimeError("Could not find MUUFL labels in the provided file.")

    labels = np.array(labels)
    if new_size is not None:
        labels = resize_image_nn(labels, new_size)
        labels = np.round(labels).astype(np.int32)

    return labels


def visualize_gt(labels: np.ndarray, save_path: str, c, title):
    """Visualize MUUFL ground truth using the same colormap as segmentation (turbo).

    - labels: 2D array, with -1 meaning unlabeled. Values > 0 are classes.
    - c: total number of classes for color scaling (e.g., 11). If None, use labels' max.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Prepare a masked array to show unlabeled as black
    labels_vis = labels.copy()
    labels_vis = labels_vis.astype(float)
    labels_vis[labels_vis < 0] = np.nan

    vmax = int(np.nanmax(labels_vis)) if c is None else int(c)
    vmin = 1  # common convention in current code

    plt.figure(figsize=(10, 8))
    im = plt.imshow(labels_vis, cmap='turbo', vmin=vmin, vmax=vmax)
    plt.title(title or 'MUUFL Ground Truth (turbo colormap)')
    plt.axis('off')
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Class ID')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Default settings aligned with the resized MUUFL pipeline
    mat_path = r'D:\mpfcm_8_6_12 (1)\mpfcm_8_6_12\data\muufl_gulfport_campus_1_hsi_220_label.mat'
    # Use the same new_size used in muulf_segmentation_resized (new_size2)
    new_size = (110, 164)  # (width, height)
    total_classes = 11     # default MUUFL classes

    print("Loading MUUFL ground truth...")
    labels = load_muufl_labels(mat_path, new_size=new_size)
    print(f"Labels shape (resized): {labels.shape}")

    # Add +1 to non-negative labels, keep -1 as unlabeled
    labels_plus = labels.copy()
    labels_plus += 1

    print("Visualizing ground truth (+1 labels) with 'turbo' colormap...")
    visualize_gt(labels_plus, 'results/muufl_gt_resized_plus1.png', c=None, title='MUUFL Ground Truth (Resized, +1)')
    print("Saved: results/muufl_gt_resized_plus1.png")


if __name__ == '__main__':
    main()


