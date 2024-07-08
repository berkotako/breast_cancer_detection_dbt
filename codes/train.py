import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import monai
from monai.data import ImageDataset, DataLoader
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity, RepeatChannel
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score, roc_curve, confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision.models.video import r3d_18

def load_image_paths_and_labels(cancer_dir, normal_dir):
    """
    Load image paths and corresponding labels from the cancer and normal directories.

    Args:
    - cancer_dir (str): Directory containing the cancer image files.
    - normal_dir (str): Directory containing the normal image files.

    Returns:
    - image_paths (List[str]): List of image file paths.
    - labels (List[int]): List of labels corresponding to the image file paths.
    """
    image_paths = []
    labels = []

    # Load cancer images
    for file in os.listdir(cancer_dir):
        if file.endswith('.nii.gz'):
            image_paths.append(os.path.join(cancer_dir, file))
            labels.append(1)  # Cancer label

    # Load normal images
    for file in os.listdir(normal_dir):
        if file.endswith('.nii.gz'):
            image_paths.append(os.path.join(normal_dir, file))
            labels.append(0)  # Normal label

    return image_paths, labels

def plot_and_save_metrics(epochs, train_metrics, val_metrics, metric_name, save_path):
    plt.figure()
    plt.plot(epochs, train_metrics, label=f'Train {metric_name}')
    plt.plot(epochs, val_metrics, label=f'Validation {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(f'Train and Validation {metric_name}')
    plt.savefig(os.path.join(save_path, f'{metric_name}.png'))

class ResNeXt3D(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNeXt3D, self).__init__()
        self.model = r3d_18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Define data paths
    cancer_dir = '/content/drive/MyDrive/Praktikum/common_datasets/common_30P/'  # Directory containing the cancer image files
    normal_dir = '/content/drive/MyDrive/Praktikum/common_datasets/common_30P_normal2/'  # Directory containing the normal image files

    # Number of epochs
    num_epochs = 50  # Change this value to set the number of epochs

    # Load image paths and labels
    images, labels = load_image_paths_and_labels(cancer_dir, normal_dir)
    labels = np.array(labels, dtype=np.int64)

    # Debugging: Print number of images and labels
    print(f"Number of images: {len(images)}, Number of labels: {len(labels)}")
    print(f"Labels distribution: {np.bincount(labels)}")

    # Define transforms (Resize -> 96,96,96)
    train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), RepeatChannel(repeats=3), Resize((96, 96, 32)), RandRotate90()])
    val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), RepeatChannel(repeats=3), Resize((96, 96, 32))])

    # Split dataset into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Debugging: Print train and validation label distributions
    print(f"Train labels distribution: {np.bincount(train_labels)}")
    print(f"Validation labels distribution: {np.bincount(val_labels)}")

    # Create datasets and dataloaders
    train_ds = ImageDataset(image_files=train_images, labels=train_labels, transform=train_transforms)
    val_ds = ImageDataset(image_files=val_images, labels=val_labels, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=2, pin_memory=torch.cuda.is_available())

    # Create ResNeXt50 3D, CrossEntropyLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNeXt3D(num_classes=2).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)

    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    epoch_accuracy_values = list()
    epoch_recall_values = list()
    epoch_auc_values = list()
    epoch_f1_values = list()
    epoch_mcc_values = list()
    writer = SummaryWriter()

    for epoch in range(num_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{num_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_preds = []
            val_true = []
            with torch.no_grad():
                num_correct = 0.0
                metric_count = 0
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    val_outputs = model(val_images)
                    val_preds.extend(val_outputs.argmax(dim=1).cpu().numpy())
                    val_true.extend(val_labels.cpu().numpy())
                    value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                    metric_count += len(value)
                    num_correct += value.sum().item()
            accuracy = accuracy_score(val_true, val_preds)
            recall = recall_score(val_true, val_preds)
            auc = roc_auc_score(val_true, val_preds)
            f1 = f1_score(val_true, val_preds)
            mcc = matthews_corrcoef(val_true, val_preds)

            epoch_accuracy_values.append(accuracy)
            epoch_recall_values.append(recall)
            epoch_auc_values.append(auc)
            epoch_f1_values.append(f1)
            epoch_mcc_values.append(mcc)

            if accuracy > best_metric:
                best_metric = accuracy
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model_classification3d_5E_RNext50_3D_512_20P.pth")
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current accuracy: {accuracy:.4f} best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}"
            )
            writer.add_scalar("val_accuracy", accuracy, epoch + 1)

    # Save metrics as PNG
    epochs = list(range(1, num_epochs + 1))
    metrics = {
        'accuracy': epoch_accuracy_values,
        'recall': epoch_recall_values,
        'auc': epoch_auc_values,
        'f1': epoch_f1_values,
        'mcc': epoch_mcc_values
    }
    for metric_name, metric_values in metrics.items():
        plot_and_save_metrics(epochs[:len(metric_values)], epoch_loss_values[:len(metric_values)], metric_values, metric_name, '/content/')

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()

if __name__ == "__main__":
    main()
