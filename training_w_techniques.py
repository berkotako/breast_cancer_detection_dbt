import os
import math
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# Load the CSV file
csv_file_path = '/Users/berkk/Desktop/train_phase2/labels.csv'
data = pd.read_csv(csv_file_path)

# Check for missing values in the 'Class' column
if data['Class'].isnull().sum() > 0:
    print("Warning: Missing values found in 'Class' column. Dropping rows with missing labels.")
    data = data.dropna(subset=['Class'])

# Identify unique class labels
unique_classes = data['Class'].unique()
print(f"Unique class labels before mapping: {unique_classes}")

# Convert class labels to binary (benign=0, cancer=1)
label_map = {'benign': 0, 'cancer': 1}

# Apply the mapping, any class not in the label_map will become NaN
data['Class'] = data['Class'].map(label_map)

# Check for NaN values after mapping
if data['Class'].isnull().sum() > 0:
    print("Error: NaN values found in 'Class' column after mapping. The following labels were not mapped correctly:")
    unmapped_classes = data[data['Class'].isnull()]['Class'].unique()
    print(unmapped_classes)
    # Drop rows with unmapped class labels
    data = data.dropna(subset=['Class'])
else:
    print("No NaN values in 'Class' column after mapping.")

# Define your custom dataset
class BreastCancerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Base directory where images are stored
base_dir = 'C:\\Users\\berkk\\Desktop\\train_phase2\\patch_100_512_2\\'

# Function to construct image paths based on directory structure
def construct_image_paths_and_labels(row, base_dir, technique):
    pattern = re.compile(
        f"{row['PatientID']}_{row['StudyUID']}_{row['View']}_{row['Slice']}_grey_\\d+.png"
    )
    technique_dir = os.path.join(base_dir, technique, row['PatientID'])
    image_paths = []
    if os.path.exists(technique_dir):
        for filename in os.listdir(technique_dir):
            if pattern.match(filename):
                img_path = os.path.join(technique_dir, filename)
                image_paths.append((img_path, row['Class']))
    return image_paths

# Training function
def train_model(model, criterion, optimizer, dataloaders, num_epochs=25, model_name='', technique=''):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    metrics = {
        'ModelName': [],
        'PatchName': [],
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_confusion_matrix': [],
        'val_auc': [],
        'val_roc_curve': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        metrics['ModelName'].append(model_name)
        metrics['PatchName'].append(technique)
        metrics['epoch'].append(epoch)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            all_labels = []
            all_preds = []

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float()

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.sigmoid(outputs)
                    preds = preds > 0.5
                    loss = criterion(outputs, labels.unsqueeze(1))

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data.unsqueeze(1))

                # Collect all labels and predictions for metrics
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Calculate additional metrics
            all_labels = np.array(all_labels)
            all_preds = np.array(all_preds)
            cm = confusion_matrix(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds)
            recall = recall_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)
            auc_score = roc_auc_score(all_labels, all_preds)
            fpr, tpr, _ = roc_curve(all_labels, all_preds)

            if phase == 'train':
                metrics['train_loss'].append(epoch_loss)
                metrics['train_acc'].append(epoch_acc.item())
            else:
                metrics['val_loss'].append(epoch_loss)
                metrics['val_acc'].append(epoch_acc.item())
                metrics['val_precision'].append(precision)
                metrics['val_recall'].append(recall)
                metrics['val_f1'].append(f1)
                metrics['val_confusion_matrix'].append(cm.tolist())
                metrics['val_auc'].append(auc_score)
                metrics['val_roc_curve'].append((fpr.tolist(), tpr.tolist()))

            print(f'{phase} Confusion Matrix:\n{cm}')
            print(f'{phase} Precision: {precision:.4f}')
            print(f'{phase} Recall: {recall:.4f}')
            print(f'{phase} F1 Score: {f1:.4f}')
            print(f'{phase} AUC: {auc_score:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_csv_path = f'C:\\Users\\berkk\\Desktop\\models_w_512_2pngs_50E\\metrics_2_{technique}.csv'
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f'Metrics saved for technique {technique} at {metrics_csv_path}')

    # Plot metrics
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(metrics['epoch'], metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['epoch'], metrics['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Loss for {technique}')

    plt.subplot(1, 3, 2)
    plt.plot(metrics['epoch'], metrics['train_acc'], label='Train Acc')
    plt.plot(metrics['epoch'], metrics['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Accuracy for {technique}')

    plt.subplot(1, 3, 3)
    plt.plot(metrics['epoch'], metrics['val_precision'], label='Precision')
    plt.plot(metrics['epoch'], metrics['val_recall'], label='Recall')
    plt.plot(metrics['epoch'], metrics['val_f1'], label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title(f'Precision/Recall/F1 for {technique}')

    plt.tight_layout()
    plt.savefig(f'C:\\Users\\berkk\\Desktop\\models_w_512_2pngs_50E\\figures\\metrics_2_plot_{technique}.png')
    plt.show()

    # Plot ROC Curve
    plt.figure()
    for fpr, tpr in metrics['val_roc_curve']:
        plt.plot(fpr, tpr, label=f'ROC Curve (area = {auc(fpr, tpr):.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for {technique}')
    plt.legend(loc="lower right")
    plt.savefig(f'C:\\Users\\berkk\\Desktop\\models_w_512_2pngs_50E\\figures\\roc_curve_{technique}.png')
    plt.show()

    return model

# Data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Training configurations
#techniques = ['sliding_window', 'random_sampling', 'grid_sampling', 'overlap_tiling', 'contour_based_sampling']
techniques = ['contour_based_sampling']
num_epochs = 50


# ResNeXt model definition
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        D = int(math.floor(planes * (baseWidth / 64)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D*C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(D*C)
        self.conv2 = nn.Conv2d(D*C, D*C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D*C)
        self.conv3 = nn.Conv2d(D*C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNeXt(nn.Module):
    def __init__(self, baseWidth, cardinality, layers, num_classes):
        super(ResNeXt, self).__init__()
        block = Bottleneck

        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.num_classes = num_classes
        self.inplanes = 64
        self.output_size = 64

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnext50(baseWidth, cardinality, num_classes):
    model = ResNeXt(baseWidth, cardinality, [3, 4, 6, 3], num_classes)
    return model

# Iterate over each technique for training
for technique in techniques:
    print(f'Training with technique: {technique}')

    # Create image paths and labels for the current technique
    image_paths_labels = data.apply(construct_image_paths_and_labels, base_dir=base_dir, technique=technique, axis=1)
    image_paths_labels = [item for sublist in image_paths_labels for item in sublist]

    # Ensure image_paths_labels is not empty
    if len(image_paths_labels) == 0:
        print(f"Error: No valid image paths found for technique {technique}. Check the directory structure and file naming.")
        continue

    # Split the paths and labels into separate lists
    image_paths, labels = zip(*image_paths_labels)

    # Split the dataset into training and validation sets
    train_image_paths, val_image_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Create datasets and dataloaders for the current technique
    train_dataset = BreastCancerDataset(train_image_paths, train_labels, transform=data_transforms['train'])
    val_dataset = BreastCancerDataset(val_image_paths, val_labels, transform=data_transforms['val'])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=False)
    }

    # Initialize the model for each technique
    model = resnext50(baseWidth=4, cardinality=32, num_classes=1)  # num_classes=1 for binary classification
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function and optimizer for each technique
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model

    model = train_model(model, criterion, optimizer, dataloaders, num_epochs=num_epochs, model_name='ResNeXt50', technique=technique)

    # Save the model for each technique
    model_save_path = f'C:\\Users\\berkk\\Desktop\\models_w_512_2pngs_50E\\resnext50_2_model_{technique}.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved for technique {technique} at {model_save_path}')
