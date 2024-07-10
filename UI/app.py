from flask import Flask, request, jsonify, render_template
import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from monai.transforms import Compose, EnsureChannelFirst, ScaleIntensity, Resize, LoadImage, RepeatChannel
from monai.data import MetaTensor

app = Flask(__name__)

# Path to the PyTorch model
MODEL_PATH = 'C:/Users/berkk/Desktop/TUM/Praktikum/UI/models/best_met_20P.pth'

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your model architecture here, using r3d_18 from torchvision
class ResNeXt3D(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(ResNeXt3D, self).__init__()
        # Load the pretrained r3d_18 model
        self.model = r3d_18(pretrained=True)
        # Replace the fully connected layer to match the number of classes
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Function to load the model from its saved state
def load_model(model_path):
    model = ResNeXt3D(num_classes=2)  # Adjust num_classes if different
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Image transformations, including loading and preprocessing
transforms = Compose([
    LoadImage(image_only=True),  # Automatically determines the correct reader
    ScaleIntensity(),
    EnsureChannelFirst(),
    RepeatChannel(repeats=3),
    Resize((96, 96, 32))  # Adjust dimensions if needed
])

# Function to preprocess a single image
def process_single_image(file_path, transforms):
    image_tensor = transforms(file_path)
    return image_tensor.unsqueeze(0).to(device)  # Add batch dimension and send to device

# Function to test the model on a single image
def test_single_image(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()
    
    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_nifti_file():
    print("Received file upload")
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file
    nifti_file_path = os.path.join('uploads', file.filename)
    file.save(nifti_file_path)
    print("Saved uploaded file:", nifti_file_path)

    # Load the NIfTI image
    # nifti_img = nib.load(nifti_file_path)
    # img_data = nifti_img.get_fdata()
    # print("Loaded NIfTI image with shape:", img_data.shape)

    #Run inference on single patch
    # Load the model
    model = load_model(model_path=MODEL_PATH)

    # Load and preprocess the image
    image_tensor = process_single_image(nifti_file_path, transforms)

    # Test the model on the single image
    prediction = test_single_image(model, image_tensor)
    print(f"Prediction: {prediction}")  # Outputs the class index
    return jsonify({'result': 'Cancer' if prediction == 1 else 'Normal'})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
