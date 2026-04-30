import cv2
import numpy as np
import sys
import os

def filter_by_pixel_count_and_binarize(input_path, output_path, min_size=50):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {input_path}")
        sys.exit(1)

    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 1. Remove small white chunks (background noise)
    # Finds all connected white pixels
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    
    cleaned_white = np.zeros_like(binary_img)
    for i in range(1, num_labels): # Skip label 0 (background)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned_white[labels == i] = 255

    # 2. Remove small black chunks (holes inside fibers)
    # Invert the image to find black connected components
    inverted = cv2.bitwise_not(cleaned_white)
    num_labels_inv, labels_inv, stats_inv, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    
    final_img = cleaned_white.copy()
    for i in range(1, num_labels_inv): # Skip label 0 (which is now the white fibers)
        if stats_inv[i, cv2.CC_STAT_AREA] < min_size:
            # Fill the small black hole with white
            final_img[labels_inv == i] = 255

    for y in range(final_img.shape[0]):
        for x in range(final_img.shape[1]):
            final_img[y, x] = 255 if final_img[y, x] >= 128 else 0

    cv2.imwrite(output_path, final_img)

if __name__ == "__main__":    
    for filename in os.listdir("data/manual_masks/"):
        input_file = os.path.join("data/manual_masks/", filename)
        output_file = os.path.join("data/manual_masks/", filename)
        min_pixel_size = 10
        filter_by_pixel_count_and_binarize(input_file, output_file, min_pixel_size)
