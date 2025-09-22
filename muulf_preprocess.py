from pymatreader import read_mat
import cv2
import numpy as np

def load_and_crop_rgb():
    # Load the mat file
    data = read_mat("C:/Users/Admin/scikit_learn_data/muufl/MUUFLGulfport-0.1/MUUFLGulfportDataCollection/muufl_gulfport_campus_w_lidar_1.mat")
    
    # Extract RGB data
    rgb = data["hsi"]["RGB"]
    print(f"Original RGB shape: {rgb.shape}")
    
    # Crop the leftmost 220 pixels
    cropped_rgb = rgb[:, :220, :]
    print(f"Cropped RGB shape: {cropped_rgb.shape}")
    
    return cropped_rgb * 255  # Scale to 0-255 range

if __name__ == "__main__":
    # Test the function
    cropped = load_and_crop_rgb()
    
    # Save the cropped image
    cv2.imwrite("cropped_rgb.png", cv2.cvtColor(cropped.astype(np.uint8), cv2.COLOR_RGB2BGR))
