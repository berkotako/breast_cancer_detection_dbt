from flask import Flask, request, jsonify, render_template
import os
import nibabel as nib
import pydicom
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from monai.transforms import Compose, EnsureChannelFirst, ScaleIntensity, Resize, LoadImage, RepeatChannel
import cv2

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

#Get dicom file path, read it, in memory get the middle Z dimesion, find contours, get the bounding box, crop the image, return the middle point of the bounding box
import pydicom

def get_middle_point(dicom_file_path):
    dicom_img = pydicom.dcmread(dicom_file_path)
    img_data = dicom_img.pixel_array.astype(float)
    print("Loaded DICOM image with shape:", img_data.shape)
    middle_z = img_data.shape[0] // 2
    middle_slice = img_data[middle_z]
    middle_slice = middle_slice / np.max(middle_slice) * 255
    middle_slice = middle_slice.astype(np.uint8)
    middle_slice = middle_slice - np.min(middle_slice)
    _, edges = cv2.threshold(middle_slice, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.erode(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(contours[0])
    print("Bounding box:", x, y, w, h)
    #img_result = middle_slice[y:y+h, x:x+w]
    middle_point = (x + w // 2, y + h // 2)
    #shows the 500x500 patch around the middle point
    #Show the 500x500 patch around the middle point
    print("Middle point:", middle_point)
    print("Middle slice shape:", middle_slice.shape)
    #rectengle around the middle point
    patch = cv2.rectangle(middle_slice, (middle_point[0], middle_point[1] - 500), (middle_point[0]+500, middle_point[1]), (255, 0, 0), 2)
    #Resize the image 0.5
    patch = cv2.resize(patch, (patch.shape[1] // 2, patch.shape[0] // 2))
    cv2.imshow("Patch", patch)
    cv2.waitKey(0)
    return middle_point


def create_3d_patch(dicom_file_path, middle_point, patch_size=(20, 500, 500)):
    # Read the DICOM file using pydicom
    dicom_img = pydicom.dcmread(dicom_file_path)
    # Assuming the DICOM image is loaded as a 3D dataset
    img_data = dicom_img.pixel_array
    print("Loaded DICOM image with shape:", img_data.shape)

    # Unpack the middle point
    x, y = middle_point
    middle_z = int(img_data.shape[0] // 2)
    # Define the patch size
    patch_size_z, patch_size_x, patch_size_y = patch_size
    print("x:", x, "y:", y, "middle_z:", middle_z)
    # Define the patch list
    patches = []
    paths = []

    patch_1 = img_data[middle_z - patch_size_z // 2:middle_z + patch_size_z // 2, x:x + patch_size_x, y:y + patch_size_y]
    print("Patch 1 shape:", patch_1.shape)
    patch_2 = img_data[middle_z - patch_size_z // 2:middle_z + patch_size_z // 2, x - patch_size_x:x, y:y + patch_size_y]
    print("Patch 2 shape:", patch_2.shape)
    patch_3 = img_data[middle_z - patch_size_z // 2:middle_z + patch_size_z // 2, x:x + patch_size_x, y - patch_size_y:y]
    print("Patch 3 shape:", patch_3.shape)
    patch_4 = img_data[middle_z - patch_size_z // 2:middle_z + patch_size_z // 2, x - patch_size_x:x, y - patch_size_y:y]
    print("Patch 4 shape:", patch_4.shape)
    print("Patches created successfully")
    # Append the patches to the list
    patches.append(patch_1)
    patches.append(patch_2)
    patches.append(patch_3)
    patches.append(patch_4)
    
    #Save the patches
    for i, patch in enumerate(patches):
        patch_nifti = nib.Nifti1Image(patch, np.eye(4))
        patch_path = os.path.join('uploads', f'patch_{i}.nii.gz')
        nib.save(patch_nifti, patch_path)
        paths.append(patch_path)
    return paths






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

    #Run inference on single patch
    # Load the model
    model = load_model(model_path=MODEL_PATH)
    #Create patches
    patches = create_3d_patch(nifti_file_path, get_middle_point(nifti_file_path))
    predictions = []
    #Run process_single_image for each patch
    for patch in patches:
        patch_tensor = process_single_image(patch, transforms)
        prediction = test_single_image(model, patch_tensor)
        predictions.append(prediction)
    print(f"Prediction: {predictions}")  # Outputs the class index
    return jsonify({'result': 'Cancer' if prediction == 1 else 'Normal'})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
