import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Function to preprocess the image
def preprocess_image(image):
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(normalized_image)
    return enhanced_image

# Function to detect contours and extract regions of interest
def detect_contours(image):
    _, thresholded_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Function to check if a patch is mostly black
def is_mostly_black(patch, threshold=0.2):
    black_pixels = np.sum(patch == 0)
    total_pixels = patch.size
    return (black_pixels / total_pixels) >= threshold

# Function to extract patches using sliding window technique
def sliding_window(image, patch_size, stride):
    patches = []
    for y in range(0, image.shape[0] - patch_size + 1, stride):
        for x in range(0, image.shape[1] - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            if not is_mostly_black(patch):
                patches.append(patch)
                if len(patches) >= 100:
                    return patches
    return patches

# Function to extract patches using random sampling
def random_sampling(image, patch_size, num_patches):
    patches = []
    attempts = 0
    while len(patches) < num_patches and attempts < num_patches * 10:  # Avoid infinite loop
        x = np.random.randint(0, image.shape[1] - patch_size)
        y = np.random.randint(0, image.shape[0] - patch_size)
        patch = image[y:y+patch_size, x:x+patch_size]
        if patch.shape == (patch_size, patch_size) and not is_mostly_black(patch):
            patches.append(patch)
        attempts += 1
    return patches

# Function to extract patches using grid-based sampling
def grid_sampling(image, grid_size, patch_size):
    patches = []
    grid_height = image.shape[0] // grid_size
    grid_width = image.shape[1] // grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            start_y = i * grid_height
            start_x = j * grid_width
            patch = image[start_y:start_y+patch_size, start_x:start_x+patch_size]
            if patch.shape == (patch_size, patch_size) and not is_mostly_black(patch):
                patches.append(patch)
            if len(patches) >= 100:
                return patches
    return patches

# Function to extract patches using overlap tiling
def overlap_tiling(image, patch_size, overlap):
    stride = patch_size - overlap
    return sliding_window(image, patch_size, stride)

# Function to extract patches using contour-based technique
def contour_based_sampling(image, patch_size, num_patches):
    patches = []
    contours = detect_contours(image)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        for i in range(y, y + h, patch_size):
            for j in range(x, x + w, patch_size):
                patch = image[i:i+patch_size, j:j+patch_size]
                if patch.shape == (patch_size, patch_size) and not is_mostly_black(patch):
                    patches.append(patch)
                if len(patches) >= 100:
                    return patches
    return patches

# Function to save patches with the specified naming convention
def save_patches(patches, technique, patient_id, study_uid, view, slice_num, output_dir):
    technique_output_dir = os.path.join(output_dir, technique, patient_id)
    os.makedirs(technique_output_dir, exist_ok=True)
    
    for idx, patch in enumerate(patches):
        img_name = f"{patient_id}_{study_uid}_{view}_{slice_num}_grey_{idx}.png"
        cv2.imwrite(os.path.join(technique_output_dir, img_name), patch)

# Main function to process all images in the directory
def process_directory(input_dir, output_dir, patch_size=128, num_patches=10, grid_size=4, overlap=32, stride=64):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.png'):
                # Extracting metadata from the filename
                parts = file.split('_')
                patient_id = parts[0]
                study_uid = parts[1]
                view = parts[2]
                slice_num = parts[3].split('.')[0]  # Assuming slice number is before the extension

                image_path = os.path.join(root, file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                preprocessed_image = preprocess_image(image)
                contours = detect_contours(preprocessed_image)

                patches_sliding = sliding_window(preprocessed_image, patch_size, stride)
                save_patches(patches_sliding, 'sliding_window', patient_id, study_uid, view, slice_num, output_dir)
                
                patches_random = random_sampling(preprocessed_image, patch_size, num_patches)
                save_patches(patches_random, 'random_sampling', patient_id, study_uid, view, slice_num, output_dir)
                
                patches_grid = grid_sampling(preprocessed_image, grid_size, patch_size)
                save_patches(patches_grid, 'grid_sampling', patient_id, study_uid, view, slice_num, output_dir)
                
                patches_overlap = overlap_tiling(preprocessed_image, patch_size, overlap)
                save_patches(patches_overlap, 'overlap_tiling', patient_id, study_uid, view, slice_num, output_dir)
                
                patches_contour = contour_based_sampling(preprocessed_image, patch_size, num_patches)
                save_patches(patches_contour, 'contour_based_sampling', patient_id, study_uid, view, slice_num, output_dir)


# Specify input and output directories
input_directory = 'C:\\Users\\berkk\\Desktop\\train_phase2\\grey_original\\'
output_directory = 'C:\\Users\\berkk\\Desktop\\train_phase2\\patch_100\\'

# Process the directory
process_directory(input_directory, output_directory)
