from flask import Flask, request, jsonify, render_template
import os
import pydicom
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import r3d_18
from monai.transforms import EnsureChannelFirst, Compose, Resize, ScaleIntensity, RepeatChannel
from monai.data import MetaTensor
import pydicom
import nibabel as nib

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

# Define transforms
transforms = Compose([
    ScaleIntensity(),
    EnsureChannelFirst(),
    RepeatChannel(repeats=3),  # Repeat the single channel to create 3 channels
    Resize((96, 96, 96))
])

def dicom_to_nifti(dicom_file_path, output_file_path):
    # Read the DICOM file
    dicom = pydicom.dcmread(dicom_file_path)
    # Get the pixel data from the DICOM file
    pixel_array = dicom.pixel_array
    # Create a NIfTI image
    nifti_img = nib.Nifti1Image(pixel_array, np.eye(4))
    # Save the NIfTI image
    nib.save(nifti_img, output_file_path)

def run_inference_on_image(nifti_file_path):
    # Load the NIfTI image
    nifti_img = nib.load(nifti_file_path)
    # Get the image data as a NumPy array
    img_data = nifti_img.get_fdata()
    # Ensure image data is in the shape (C, H, W, D)
    img_data = np.expand_dims(img_data, axis=0)
    
    # Convert the image data to a MetaTensor and set the metadata
    img_meta = MetaTensor(img_data, meta={'original_channel_dim': -1})
    # Apply transforms
    img_transformed = transforms(img_meta)
    # Extract the numpy array from the MetaTensor
    img_np = img_transformed.numpy()
    # Convert the image to a PyTorch tensor
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device).float()
    # Run inference
    with torch.no_grad():
        output = model(img_tensor)
    return output.cpu().numpy()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file
    dicom_file_path = os.path.join('uploads', file.filename)
    file.save(dicom_file_path)

    # Convert DICOM to NIfTI
    nifti_file_path = dicom_file_path.replace('.dcm', '.nii.gz')
    dicom_to_nifti(dicom_file_path, nifti_file_path)

    # Run inference on the NIfTI file
    output = run_inference_on_image(nifti_file_path)
    # Assuming the output is a 2-element array [probability of normal, probability of cancer]
    cancer_probability = output[0][1]
    result = 'Cancer' if cancer_probability > 0.5 else 'Normal'

    return jsonify({'result': result, 'cancer_probability': float(cancer_probability)})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)