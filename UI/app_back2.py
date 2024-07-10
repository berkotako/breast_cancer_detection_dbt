from flask import Flask, request, jsonify, render_template
import os
import pydicom
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from monai.transforms import EnsureChannelFirst, Compose, Resize, ScaleIntensity, RepeatChannel
from monai.data import MetaTensor
import random


app = Flask(__name__)

# Path to the PyTorch model
MODEL_PATH = 'C:/Users/berkk/Desktop/TUM/Praktikum/UI/models/best_met_20P.pth'

# Define the ResNeXt3D model structure
class ResNeXt3D(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNeXt3D, self).__init__()
        self.model = r3d_18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Load the PyTorch model
model = ResNeXt3D(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define transformsEnsureChannelFirst(),RepeatChannel(repeats=3),  # Repeat the single channel to create 3 channels
transforms = Compose([
    ScaleIntensity(),
    EnsureChannelFirst(),
    RepeatChannel(repeats=3),  # Repeat the single channel to create 3 channels
    Resize((96, 96, 96))
])
val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), RepeatChannel(repeats=3), Resize((96, 96, 96))])


def dicom_to_nifti(dicom_file_path, output_file_path):
    print("Converting DICOM to NIfTI")
    # Read the DICOM file
    dicom = pydicom.dcmread(dicom_file_path)
    # Get the pixel data from the DICOM file
    pixel_array = dicom.pixel_array
    print("Shape of pixel array:", pixel_array.shape)
     # Ensure the pixel array is in the correct format, <z, x, y>
    if pixel_array.ndim == 3:
        # If the data is in <x, y, z> format, we need to move the last axis to the first
        pixel_array = np.moveaxis(pixel_array, 1, 2)
    #Rotate the image 180 degrees
    pixel_array = np.rot90(pixel_array, 2)
    #Reverse the image alon z axis
    pixel_array = np.flip(pixel_array, axis=2)
    # Create a NIfTI image
    nifti_img = nib.Nifti1Image(pixel_array, np.eye(4))
    #Normalize the image
    nifti_img = nib.Nifti1Image(nifti_img.get_fdata()/np.max(nifti_img.get_fdata()), nifti_img.affine)
    # Save the NIfTI image
    print("Shape of NIfTI image:", nifti_img.shape)
    nib.save(nifti_img, output_file_path)
    print("Converted DICOM to NIfTI:", output_file_path)


def extract_patches(img_data, patch_size_z=20, min_dim=200, max_dim=250):
    # 0 -> Z dimension, 2 -> X dimension, 1 -> Y dimension
    print("Extracting patches")
    z_dim = img_data.shape[0]
    patches = []
    patch_count = 0
    print("Image shape:", img_data.shape)
    for z in range(0, z_dim, patch_size_z):
        z_end = min(z + patch_size_z, z_dim)
        if z_end - z < patch_size_z:
            continue  # Skip if the patch depth is less than the specified patch size

        x_dim = img_data.shape[2]
        y_dim = img_data.shape[1]
        x_start_points = []
        y_start_points = []
        print("Extracting patches at z:", z, "-", z_end)
        while len(x_start_points) < 3 and len(y_start_points) < 3:  # Trying to create 3 patches
            print("Trying to extract patch at z:", z)
            height = random.randint(min_dim, max_dim)
            width = random.randint(min_dim, max_dim)
            #calc x_dim - height and y_dim - width to avoid out of bounds, if these are negative, continue
            if x_dim - height < 0 or y_dim - width < 0:
                x_start = int(x_dim/2)
                y_start = int(y_dim/2)
            else:
                x_start = random.randint(0, x_dim - height)
                y_start = random.randint(0, y_dim - width)
            print("Trying to extract patch at x:", x_start, "y:", y_start, "z:", z)

            if not any((abs(x_start - xs) < height and abs(y_start - ys) < width) for xs, ys in zip(x_start_points, y_start_points)):
                x_start_points.append(x_start)
                y_start_points.append(y_start)
                print("Extracting patch at x:", x_start, "y:", y_start, "z:", z)
                if y_start - width < 0 :
                    width = y_start
                elif x_start - height < 0:
                    height = x_start
                patch = img_data[z:z_end, y_start - width:y_start + width, x_start-height:x_start + height]
                print("Patch shape:", patch.shape)
                patches.append(patch)
                print("Extracted patch at x:", x_start, "y:", y_start, "z:", z)
                patch_count += 1
    print("Extracted", patch_count, "patches")
    return patches

#Save patches to a folder as nifti files
def save_patches(patches):
    for idx, patch in enumerate(patches):
        patch_nifti = nib.Nifti1Image(patch, np.eye(4))
        patch_file_path = os.path.join('uploads', f'patch_{idx}.nii.gz')
        nib.save(patch_nifti, patch_file_path)
        print("Saved patch to:", patch_file_path)

def run_inference_on_patch(patch):
    print("Running inference on patch")
    # Convert the patch to a MetaTensor and set the metadata
    patch_meta = MetaTensor(patch, meta={'original_channel_dim': 2})
    # Apply transforms
    patch_transformed = transforms(patch_meta)
    print("Transformed patch shape:", patch_transformed.shape)
    # Extract the numpy array from the MetaTensor
    patch_np = patch_transformed.numpy()
    print("Converted patch to numpy array")
    # Convert the patch to a PyTorch tensor
    patch_tensor = torch.from_numpy(patch_np).unsqueeze(0).to(device).float()
    print("Converted patch to PyTorch tensor")
    # Run inference
    with torch.no_grad():
        output = model(patch_tensor)
    print("Inference result:", output)
    return output.cpu().numpy()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Received file upload")
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file
    dicom_file_path = os.path.join('uploads', file.filename)
    file.save(dicom_file_path)
    print("Saved uploaded file:", dicom_file_path)

    # Convert DICOM to NIfTI
    nifti_file_path = dicom_file_path.replace('.dcm', '.nii.gz')
    dicom_to_nifti(dicom_file_path, nifti_file_path)
    print("Converted DICOM to NIfTI:", nifti_file_path)

    # Load the NIfTI image and extract patches
    nifti_img = nib.load(nifti_file_path)
    img_data = nifti_img.get_fdata()
    patches = extract_patches(img_data)
    print("Extracted", len(patches), "patches")

    save_patches(patches)

    # Run inference on each patch
    patch_results = [run_inference_on_patch(patch) for patch in patches]

    # Collect results for each patch
    patch_details = []
    for idx, result in enumerate(patch_results):
        cancer_probability = result[0][1]
        patch_result = {
            'patch_index': idx,
            'cancer_probability': float(cancer_probability),
            'result': 'Cancer' if cancer_probability > 0.5 else 'Normal'
        }
        patch_details.append(patch_result)

    # Aggregate results (example: average probability of cancer across patches)
    avg_cancer_probability = np.mean([result[0][1] for result in patch_results])
    final_result = 'Cancer' if avg_cancer_probability > 0.5 else 'Normal'

    return jsonify({'final_result': final_result, 'avg_cancer_probability': float(avg_cancer_probability), 'patch_details': patch_details})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
