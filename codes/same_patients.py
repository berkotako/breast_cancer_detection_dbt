import os
import shutil
from typing import List, Dict, Set
from collections import defaultdict

def collect_files_in_directories(directories: List[str]) -> Dict[str, List[str]]:
    """Collect files in each directory and map them to the directory."""
    dir_files = {}
    for directory in directories:
        files = []
        for subdir, _, file_list in os.walk(directory):
            for file in file_list:
                files.append(os.path.join(subdir, file))
        dir_files[directory] = files
    return dir_files

def extract_patient_id(file_path: str) -> str:
    """Extract patient ID from the file path assuming it is part of the filename."""
    return os.path.basename(file_path).split('_')[0]

def find_common_patient_ids(dir_files: Dict[str, List[str]]) -> Set[str]:
    """Find common patient IDs across all directories."""
    patient_ids_sets = []
    for files in dir_files.values():
        patient_ids = {extract_patient_id(file) for file in files}
        patient_ids_sets.append(patient_ids)
    common_patient_ids = set.intersection(*patient_ids_sets)
    return common_patient_ids

def copy_files_to_new_datasets(dir_files: Dict[str, List[str]], common_patient_ids: Set[str], output_dirs: List[str]):
    """Copy files to new dataset directories based on common patient IDs."""
    for directory, files in dir_files.items():
        for patient_id in common_patient_ids:
            patient_files = [file for file in files if extract_patient_id(file) == patient_id]
            for output_dir in output_dirs:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                for file in patient_files:
                    shutil.copy(file, output_dir)

# Define root directories containing the files
root_dirs = [
    '/content/drive/MyDrive/Praktikum/3D_20P_normal_2/',
    '/content/drive/MyDrive/Praktikum/3D_30P_normal_2/'
    # Add more folders as needed
]

# Define output directories where files will be copied
output_dirs = [
    '/content/3d_dataset_slices_fcsv2_20P_common_normal_2/',
    '/content/3d_dataset_slices_fcsv2_30P_common_normal_2/'
]

# Collect files in each directory
dir_files = collect_files_in_directories(root_dirs)
print(f"Collected files from {len(dir_files)} directories.")

# Find common patient IDs across directories
common_patient_ids = find_common_patient_ids(dir_files)
print(f"Found {len(common_patient_ids)} common patient IDs.")

# Copy files to new datasets based on common patient IDs
copy_files_to_new_datasets(dir_files, common_patient_ids, output_dirs)
print("Files copied to new datasets based on common patient IDs.")
