import os
import pydicom
import numpy as np
import pandas as pd
from skimage.exposure import rescale_intensity
import nibabel as nib
from tqdm import tqdm
import gc
from typing import List
import random

def load_dicom_volume(dicom_file):
    ds = pydicom.dcmread(dicom_file)
    volume = ds.pixel_array
    volume = rescale_intensity(volume, out_range=(0, 255)).astype(np.uint8)
    return volume

def get_patient_and_study_id(dicom_file_path):
    """
    Extract Patient ID and Study ID from a DICOM file.

    Parameters:
    - dicom_file_path: Path to the DICOM file.

    Returns:
    - patient_id: The Patient ID extracted from the DICOM file.
    - study_uid: The Study ID extracted from the DICOM file.
    """
    ds = pydicom.dcmread(dicom_file_path)
    patient_id = ds.PatientID
    study_uid = ds.StudyInstanceUID
    view = ds.ViewPosition
    return patient_id, study_uid, view

def collect_dicom_files(root_dir: str) -> List[str]:
    """Collect all DICOM file paths from the directory structure."""
    dicom_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.dcm'):
                dicom_files.append(os.path.join(subdir, file))
    return dicom_files

def save_slices_based_on_csv(csv_path, dicom_files, save_dir):
    # Load the CSV file
    csv_data = pd.read_csv(csv_path)

    # Create a DataFrame to save metadata
    df = pd.DataFrame(columns=['PatientID', 'StudyUID', 'view', 'img_path', 'Normal', 'Actionable', 'Benign', 'Cancer'])

    os.makedirs(save_dir, exist_ok=True)

    for dicom_file in tqdm(dicom_files):
        try:
            patient_id, study_uid = get_patient_and_study_id(dicom_file)
            matching_rows = csv_data[(csv_data['PatientID'] == patient_id)]

            if not matching_rows.empty:
                volume = load_dicom_volume(dicom_file)

                for _, row in matching_rows.iterrows():
                    view = row['View']
                    x = row['X']
                    y = row['Y']
                    width = row['Width']
                    height = row['Height']
                    slice_idx = row['Slice']

                    # Ensure the slice index is within bounds
                    if 0 <= slice_idx < volume.shape[0]:
                      try:
                        slice_volume = volume[slice_idx-10:slice_idx+10,y-height-128:y+height+128 ,x-width-128:x+width+128]
                        slice_name = f"{patient_id}_{study_uid}_{view}_slice_{slice_idx}.nii.gz"
                        slice_nifti = nib.Nifti1Image(slice_volume, np.eye(4))
                        nib.save(slice_nifti, os.path.join(save_dir, slice_name))

                        df.loc[len(df)] = [
                            patient_id, study_uid, view, os.path.join(save_dir, slice_name),
                            row['Normal'], row['Actionable'], row['Benign'], row['Cancer']
                        ]
                      except Exception as a:
                        print(print(f"Error getting volume {dicom_file}: {a}"))
        except Exception as e:
            print(f"Error processing {dicom_file}: {e}")
        gc.collect()

    # Save the metadata DataFrame
    df.to_csv(os.path.join(save_dir, 'slices_metadata.csv'), index=False)

# Define paths
csv_path = '/content/drive/MyDrive/Praktikum/boxes.csv'
dicom_dir = '/content/dataset/Breast-Cancer-Screening-DBT/normal/'
save_dir = '/content/3d_slices_nii_3D_30P_normal/'

# Collect all DICOM file paths
dicom_files = collect_dicom_files(dicom_dir)
print(f"Found {len(dicom_files)} DICOM files.")
print(dicom_files[0])

# Process the DICOM files based on the CSV data
save_slices_based_on_csv(csv_path, dicom_files, save_dir)

