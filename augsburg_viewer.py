import numpy as np
import scipy.io as sio
import h5py
import matplotlib.pyplot as plt

def read_mat(filepath):
    """Read MAT file and return its contents"""
    try:
        # First try normal scipy.io.loadmat
        data = sio.loadmat(filepath)
        return data
    except Exception as e:
        print(f"Error with standard loadmat: {e}")
        try:
            # If that fails, try h5py approach for newer MAT files
            f = h5py.File(filepath, 'r')
            data = {}
            for k, v in f.items():
                data[k] = np.array(v)
            return data
        except Exception as e2:
            print(f"Error loading MAT file with h5py: {e2}")
            return None

def analyze_mat_file(filepath, name):
    print(f"\n{'='*50}")
    print(f"Analyzing {name}: {filepath}")
    print(f"{'='*50}")
    
    # Load the data
    data = read_mat(filepath)
    if data is None:
        print(f"Failed to load {filepath}")
        return
    
    # Print keys in the file
    print(f"Keys in the file: {list(data.keys())}")
    
    # Analyze each key
    for key in data.keys():
        if key.startswith('__'):  # Skip metadata keys
            continue
        
        value = data[key]
        print(f"\nKey: {key}")
        print(f"Type: {type(value)}")
        print(f"Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
        
        if isinstance(value, np.ndarray):
            # Get data type
            print(f"Data type: {value.dtype}")
            
            # Check if it's an image-like array
            if len(value.shape) >= 2:
                # Basic statistics
                unique_vals = np.unique(value)
                print(f"Number of unique values: {len(unique_vals)}")
                print(f"Min value: {value.min()}")
                print(f"Max value: {value.max()}")
                print(f"Unique values: {unique_vals[:20]}..." if len(unique_vals) > 20 else f"Unique values: {unique_vals}")
                
                # Count non-zero elements
                non_zero = np.count_nonzero(value)
                print(f"Number of non-zero elements: {non_zero} ({non_zero/value.size*100:.2f}%)")
                
                # If it's a label image, count occurrences of each label
                if len(unique_vals) < 50:  # Assume it's a label image if there are few unique values
                    print("\nLabel distribution:")
                    for label in unique_vals:
                        count = np.sum(value == label)
                        print(f"  Label {label}: {count} pixels ({count/value.size*100:.2f}%)")
                
                # Plot histogram of values
                plt.figure(figsize=(10, 6))
                plt.hist(value.flatten(), bins=min(50, len(unique_vals)), alpha=0.7)
                plt.title(f"Histogram of values in {key}")
                plt.xlabel("Value")
                plt.ylabel("Frequency")
                plt.savefig(f"{name}_{key}_histogram.png")
                
                # Visualize the data
                if len(value.shape) == 2 and len(unique_vals) < 50:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(value, cmap='viridis')
                    plt.colorbar(label='Label value')
                    plt.title(f"{key} visualization")
                    plt.savefig(f"{name}_{key}_visualization.png")
                    print(f"Saved visualization to {name}_{key}_visualization.png")
                
                plt.close('all')

# Set file path to TotalImage only
total_filepath = "D:/Download/Datasets/Datasets/HS-SAR-DSM Augsburg/TotalImage.mat"

# Analyze only TotalImage file
analyze_mat_file(total_filepath, "TotalImage")

# Detailed analysis of TotalImage
print("\n\n" + "="*50)
print("DETAILED TOTALIMAGE ANALYSIS")
print("="*50)

total_data = read_mat(total_filepath)

if total_data is not None:
    # Find the first non-metadata key in the file
    total_key = None
    
    for key in total_data.keys():
        if not key.startswith('__'):
            total_key = key
            break
    
    if total_key:
        total_labels = total_data[total_key]
        
        print(f"TotalImage shape: {total_labels.shape}")
        
        # Basic statistics
        total_pixels = total_labels.size
        labeled_pixels = np.sum(total_labels > 0)
        
        print(f"Total pixels in image: {total_pixels}")
        print(f"Labeled pixels: {labeled_pixels} ({labeled_pixels/total_pixels*100:.2f}%)")
        
        # Check label distribution
        unique_labels = np.unique(total_labels[total_labels > 0])
        print(f"\nUnique labels in TotalImage: {unique_labels}")
        print(f"Number of classes: {len(unique_labels)}")
        
        print("\nLabel distribution:")
        for label in unique_labels:
            count = np.sum(total_labels == label)
            print(f"  Label {label}: {count} pixels ({count/labeled_pixels*100:.2f}% of labeled pixels)")
        
        # Visualize TotalImage
        plt.figure(figsize=(12, 10))
        
        # Mask visualization
        plt.subplot(1, 2, 1)
        plt.imshow(total_labels > 0, cmap='gray')
        plt.title("TotalImage Mask")
        plt.axis('off')
        
        # Labels visualization
        plt.subplot(1, 2, 2)
        plt.imshow(total_labels, cmap='viridis')
        plt.colorbar(label='Label')
        plt.title("TotalImage Labels")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("totalimage_visualization.png")
        print("Saved TotalImage visualization to totalimage_visualization.png")

print("\nAnalysis complete. Check the generated images for visualizations.")