import os
import random
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import warnings
import PIL
import cv2
import pandas as pd

# Define the main data folder
MAIN_DATA_FOLDER = 'C:/Users/berkk/Desktop/BreastData/manifest-1617905855234/Breast-Cancer-Screening-DBT/'
warnings.filterwarnings("ignore")
#img = pydicom.dcmread("C:/Users/berkk/Desktop/BreastData/manifest-1617905855234/Breast-Cancer-Screening-DBT/DBT-P00003/01-01-2000-DBT-S01306-MAMMO screening digital bilateral-33603/18377.000000-NA-92351/1-1.dcm")
patient_ids = ['DBT-P00013', 'DBT-P00024', 'DBT-P00060', 'DBT-P00107', 'DBT-P00194', 'DBT-P00225', 'DBT-P00303', 'DBT-P00361', 'DBT-P00538', 'DBT-P00583', 'DBT-P00654', 'DBT-P00684', 'DBT-P00784', 'DBT-P00818', 'DBT-P00827', 'DBT-P01110', 'DBT-P01112', 'DBT-P01130', 'DBT-P01139', 'DBT-P01181', 'DBT-P01241', 'DBT-P01262', 'DBT-P01267', 'DBT-P01282', 'DBT-P01347', 'DBT-P01439', 'DBT-P01461', 'DBT-P01488', 'DBT-P01493', 'DBT-P01539', 'DBT-P01587', 'DBT-P01624', 'DBT-P01626', 'DBT-P01673', 'DBT-P01712', 'DBT-P01718', 'DBT-P01745', 'DBT-P01751', 'DBT-P01753', 'DBT-P01801', 'DBT-P01817', 'DBT-P01826', 'DBT-P01839', 'DBT-P02065', 'DBT-P02100', 'DBT-P02133', 'DBT-P02171', 'DBT-P02176', 'DBT-P02227', 'DBT-P02380', 'DBT-P02471', 'DBT-P02493', 'DBT-P02510', 'DBT-P02532', 'DBT-P02579', 'DBT-P02582', 'DBT-P02588', 'DBT-P02736', 'DBT-P02738', 'DBT-P02750', 'DBT-P02798', 'DBT-P02843', 'DBT-P02919', 'DBT-P02935', 'DBT-P03009', 'DBT-P03017', 'DBT-P03073', 'DBT-P03085', 'DBT-P03176', 'DBT-P03203', 'DBT-P03212', 'DBT-P03218', 'DBT-P03222', 'DBT-P03292', 'DBT-P03423', 'DBT-P03458', 'DBT-P03539', 'DBT-P03658', 'DBT-P03677', 'DBT-P03748', 'DBT-P03816', 'DBT-P03915', 'DBT-P03978', 'DBT-P04026', 'DBT-P04090', 'DBT-P04116', 'DBT-P04326', 'DBT-P04372', 'DBT-P04429', 'DBT-P04631', 'DBT-P04710', 'DBT-P04721', 'DBT-P04750', 'DBT-P04818', 'DBT-P04858', 'DBT-P04901', 'DBT-P05014', 'DBT-P05022', 'DBT-P05030', 'DBT-P05047', 'DBT-P05056']

df = pd.read_csv('C:\\Users\\berkk\\Desktop\\train_phase2\\BCS-DBT-labels-train.csv')

patient_ids = []
for row in df.iterrows():
    if row[1]['Cancer'] == 1:
        print(row[1]['PatientID'])
        patient_ids.append(row[1]['PatientID'])
    elif row[1]['Benign'] == 1:
        print(row[1]['PatientID'])
        patient_ids.append(row[1]['PatientID'])

# Function to get all subfolders in the main folder
def get_all_subfolders(main_folder):
    subfolders = [f.path for f in os.scandir(main_folder) if f.is_dir()]
    for subfolder in list(subfolders):
        subfolders.extend(get_all_subfolders(subfolder))
    return subfolders

# Function to get all DICOM files from the list of subfolders
def get_all_dicom_files(subfolders):
    dicom_files = []
    for subfolder in subfolders:
        for root, dirs, files in os.walk(subfolder):
            for file in files:
                if file.endswith(".dcm"):
                    dicom_files.append(os.path.join(root, file))
    #only keep the files that are in the patient_ids list
    dicom_files = [file for file in dicom_files if file.split("/")[-4] in patient_ids]
    return dicom_files

# Function to extract metadata from the file path
def extract_metadata_from_path(file_path):
    parts = file_path.split(os.sep)
    # Adjust this part based on the folder hierarchy structure
    patient_id = parts[-4]  # Assuming patient_id is 4th last
    study_uid = str(parts[-3].split("-")[3]+"-"+parts[-3].split("-")[4])  # Assuming study_uid is 3rd last
    return patient_id, study_uid

# Function to read and save images
def save_random_images(main_folder, image_count):
    subfolders = get_all_subfolders(main_folder)
    dicom_files = get_all_dicom_files(subfolders)
    dicom_files = random.sample(dicom_files, 10)
    
    for i, file in enumerate(dicom_files):
        print("Reading file:", file)
        patient_id, study_uid = extract_metadata_from_path(file)
        #if the patient id is not in the list, skip the image
        # if patient_id not in patient_ids:
        #     continue
        ds = pydicom.dcmread(file)
        img = ds.pixel_array.astype(float)
        img2 = img[random.randint(3,10)]
        #find the contours of the image and only keep the largest one and fit it to a rectangle
        #convert image ro CV_8UC1
        img2 = img2 / np.max(img2) * 255
        img2 = img2.astype(np.uint8)
        img2 = img2 - np.min(img2)
        
        _, img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img2 = cv2.erode(img2, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        x, y, w, h = cv2.boundingRect(contours[0])
        

        #show the image inside the rectangle
        img_result = img[random.randint(3,10)][y:y+h, x:x+w]   

        
        
        patient_id = patient_id.split("/")[-1]
        print("P1")
        filename = f"{patient_id}_{study_uid}_{i}.png"
        print("P2")
        save_path = "C:/Users/berkk/Desktop/BreastData/dataset_2/" + filename
        print("P3")
        plt.imsave(save_path, img_result)
        print("P4")
        print(f"Saved image: {save_path}")

# Number of images to save
IMAGE_COUNT = 100  # Change this to the desired number of images

# Execute the function
save_random_images(MAIN_DATA_FOLDER, IMAGE_COUNT)
