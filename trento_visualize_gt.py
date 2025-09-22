import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.io import loadmat
import h5py
from typing import Optional, Tuple
from scipy.optimize import linear_sum_assignment


def read_mat(filepath: str):
    try:
        return loadmat(filepath)
    except Exception:
        with h5py.File(filepath, "r") as f:
            return {k: np.array(v) for k, v in f.items()}


def resize_nn(image: np.ndarray, new_size: Optional[Tuple[int, int]]):
    if new_size is None:
        return image
    return cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)


def load_trento_labels(gt_path: str, new_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    data = read_mat(gt_path)
    key = [k for k in data.keys() if not str(k).startswith("__")][0]
    labels = np.array(data[key]).astype(np.int32)
    if new_size is not None:
        labels = resize_nn(labels, new_size)
        labels = np.round(labels).astype(np.int32)
    return labels


def visualize_gt(labels: np.ndarray, save_path: str, title: Optional[str] = None, ref_pred_path: Optional[str] = None):
    """Visualize GT with discrete palette to match segmentation mapping.

    Uses same strategy as `trento_segmentation_resized.py`: enumerate unique
    classes and assign colors from jet evenly. Zero (former -1) is shown as black
    and NOT included in the color palette to keep class colors aligned with 1..C.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    labels_int = labels.astype(int)
    height, width = labels_int.shape

    # Build color image
    color_img = np.zeros((height, width, 3), dtype=float)

    # If a reference prediction palette is provided, start from its mapping
    cluster_to_color = {}
    if ref_pred_path is not None and os.path.exists(ref_pred_path):
        try:
            pred = np.load(ref_pred_path)
            unique_pred = np.unique(pred[pred >= 0])
            # Build base palette for predicted clusters first
            colors_base = plt.cm.jet(np.linspace(0, 1, max(1, len(unique_pred))))
            cluster_to_color = {int(k): colors_base[i, :3] for i, k in enumerate(unique_pred)}
        except Exception:
            cluster_to_color = {}

    # Determine GT classes (exclude 0/1 background after +1 shift)
    gt_classes = [int(v) for v in np.unique(labels_int) if v > 1]

    if cluster_to_color:
        # Allocate colors to missing GT classes that are not in predicted palette
        # Try to align by v-1 first; if not found, append new colors
        used_colors = list(cluster_to_color.values())
        missing = []
        for v in gt_classes:
            if (v in cluster_to_color) or ((v - 1) in cluster_to_color):
                continue
            missing.append(v)

        if missing:
            extra_colors = plt.cm.jet(
                np.linspace(0, 1, len(used_colors) + len(missing))
            )[len(used_colors):]
            for idx, v in enumerate(missing):
                cluster_to_color[v] = extra_colors[idx, :3]

        # Apply colors
        for v in gt_classes:
            color = cluster_to_color.get(v, cluster_to_color.get(v - 1, None))
            if color is not None:
                color_img[labels_int == v] = color
    else:
        # Fallback: generate palette purely from GT classes
        colors = plt.cm.jet(np.linspace(0, 1, max(1, len(gt_classes))))
        for i, cls in enumerate(gt_classes):
            color_img[labels_int == cls] = colors[i, :3]

    plt.figure(figsize=(10, 8))
    plt.imshow(color_img)
    plt.title(title or 'Trento Ground Truth (matched palette)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def colorize_with_palette(label_img: np.ndarray, palette: dict[int, np.ndarray], background_values: set[int]) -> np.ndarray:
    """Return an RGB image by mapping integer labels using a fixed palette.

    background_values are rendered black.
    """
    label_int = label_img.astype(int)
    h, w = label_int.shape
    out = np.zeros((h, w, 3), dtype=float)
    for cls, color in palette.items():
        out[label_int == cls] = color
    for b in background_values:
        out[label_int == b] = (0.0, 0.0, 0.0)
    return out


def visualize_gt_and_predictions(labels_plus: np.ndarray, save_path: str):
    """Draw GT and predictions side-by-side with perfectly matched colors.

    - If adjusted and/or standard prediction npy exist, create a 1-2 rows figure:
      row for adjusted (pred_adj vs GT), row for standard (pred_u vs GT).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    pred_adj_path = 'results/trento_mpfcm_labels_full_image_resized_adj.npy'
    pred_u_path = 'results/trento_mpfcm_labels_full_image_resized.npy'

    rows = []  # list of tuples: (title_left, pred_rgb, title_right, gt_rgb)

    # Helper to build palette from a prediction
    def build_palette_from_pred(pred: np.ndarray) -> dict[int, np.ndarray]:
        unique_pred = np.unique(pred[pred >= 0])
        colors = plt.cm.jet(np.linspace(0, 1, max(1, len(unique_pred))))
        return {int(k): colors[i, :3] for i, k in enumerate(unique_pred)}

    # Adjusted row
    if os.path.exists(pred_adj_path):
        pred_adj = np.load(pred_adj_path)
        palette_adj = build_palette_from_pred(pred_adj)
        pred_adj_rgb = colorize_with_palette(pred_adj, palette_adj, background_values={-1})

        # Resize GT to prediction shape for correct overlap mapping
        labels_rs = cv2.resize(labels_plus, (pred_adj.shape[1], pred_adj.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Map GT using same palette via overlap-based assignment
        # Build contingency between pred_adj and labels_rs (>1)
        mask = (pred_adj >= 0) & (labels_rs > 1)
        gt_vals = np.array(sorted(np.unique(labels_rs[mask])))
        pred_vals = np.array(sorted(np.unique(pred_adj[mask])))
        if gt_vals.size and pred_vals.size:
            contingency = np.zeros((len(pred_vals), len(gt_vals)), dtype=np.int64)
            for i, p in enumerate(pred_vals):
                pm = pred_adj[mask] == p
                for j, g in enumerate(gt_vals):
                    contingency[i, j] = np.sum(pm & (labels_rs[mask] == g))
            r_idx, c_idx = linear_sum_assignment(-contingency)
            gt_adj_palette = {}
            for i, j in zip(r_idx, c_idx):
                pred_id = int(pred_vals[i])
                gt_id = int(gt_vals[j])
                color = palette_adj.get(pred_id)
                if color is not None:
                    gt_adj_palette[gt_id] = color
        else:
            gt_adj_palette = {}
        gt_adj_rgb = colorize_with_palette(labels_rs, gt_adj_palette, background_values={0})

        rows.append(("Prediction (Adjusted)", pred_adj_rgb, "Ground Truth (+1)", gt_adj_rgb))

    # Standard row
    if os.path.exists(pred_u_path):
        pred_u = np.load(pred_u_path)
        palette_u = build_palette_from_pred(pred_u)
        pred_u_rgb = colorize_with_palette(pred_u, palette_u, background_values={-1})

        # Resize GT to prediction shape
        labels_rs = cv2.resize(labels_plus, (pred_u.shape[1], pred_u.shape[0]), interpolation=cv2.INTER_NEAREST)

        mask = (pred_u >= 0) & (labels_rs > 1)
        gt_vals = np.array(sorted(np.unique(labels_rs[mask])))
        pred_vals = np.array(sorted(np.unique(pred_u[mask])))
        if gt_vals.size and pred_vals.size:
            contingency = np.zeros((len(pred_vals), len(gt_vals)), dtype=np.int64)
            for i, p in enumerate(pred_vals):
                pm = pred_u[mask] == p
                for j, g in enumerate(gt_vals):
                    contingency[i, j] = np.sum(pm & (labels_rs[mask] == g))
            r_idx, c_idx = linear_sum_assignment(-contingency)
            gt_u_palette = {}
            for i, j in zip(r_idx, c_idx):
                pred_id = int(pred_vals[i])
                gt_id = int(gt_vals[j])
                color = palette_u.get(pred_id)
                if color is not None:
                    gt_u_palette[gt_id] = color
        else:
            gt_u_palette = {}
        gt_u_rgb = colorize_with_palette(labels_rs, gt_u_palette, background_values={0})

        rows.append(("Prediction (u)", pred_u_rgb, "Ground Truth (+1)", gt_u_rgb))

    if not rows:
        print("No prediction files found; skipping combined visualization.")
        return

    # Draw
    plt.figure(figsize=(14, 6 * len(rows)))
    for r, (t1, img1, t2, img2) in enumerate(rows):
        plt.subplot(len(rows), 2, 2 * r + 1)
        plt.imshow(img1)
        plt.title(t1)
        plt.axis('off')

        plt.subplot(len(rows), 2, 2 * r + 2)
        plt.imshow(img2)
        plt.title(t2)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Defaults align with existing Trento scripts
    data_dir = "D:/mpfcm_8_6_12 (1)/mpfcm_8_6_12/data/Trento"
    gt_path = os.path.join(data_dir, "GT_Trento.mat")

    # No resize by default; set new_size=(width, height) if needed
    new_size = None

    print("Loading Trento ground truth...")
    labels = load_trento_labels(gt_path, new_size=new_size)
    print(f"Labels shape: {labels.shape}")

    # +1 to all labels, including -1
    labels_plus = labels + 1

    print("Visualizing ground truth (+1) with matched discrete palette...")
    # Try to match adjusted prediction palette first, then standard
    ref_adj = 'results/trento_mpfcm_labels_full_image_resized_adj.npy'
    ref_std = 'results/trento_mpfcm_labels_full_image_resized.npy'
    ref_path = ref_adj if os.path.exists(ref_adj) else (ref_std if os.path.exists(ref_std) else None)
    visualize_gt(labels_plus, 'results/trento_gt_plus1.png', title='Trento Ground Truth (+1)', ref_pred_path=ref_path)

    # Also create a combined figure GT vs predictions with identical palettes
    visualize_gt_and_predictions(labels_plus, 'results/trento_gt_pred_compare.png')
    print("Saved: results/trento_gt_pred_compare.png")
    print("Saved: results/trento_gt_plus1.png")


if __name__ == '__main__':
    main()


