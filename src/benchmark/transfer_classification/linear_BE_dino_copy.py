# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import json
from pathlib import Path
import sys
import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

from models.dino import utils
from models.dino import vision_transformer as vits

# load bigearthnet dataset
from datasets.BigEarthNet.bigearthnet_dataset_seco import Bigearthnet
from datasets.BigEarthNet.bigearthnet_dataset_seco_lmdb_s2_uint8 import LMDBDataset,random_subset

### end of change ###
import pdb
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score
import builtins
torch.device('cpu')
import torch.optim as optim

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import v2
import segmentation_models_pytorch as smp
from torchvision import transforms, models
from torchmetrics.classification import F1Score, MulticlassJaccardIndex
MEANS= {'B1': 2175.7995533938183, 'B2': 2036.2739530256324, 'B3': 2100.8864073786062, 'B4': 2259.670904983591, 'B5': 2401.4535482508154, 'B6': 3239.511063914319, 'B7': 3804.818377507816, 'B8': 3480.9846790712386, 'B8A': 4136.2014152288275, 'B9': 467.6799662569083, 'B10': 16.239072759978043, 'B11': 3768.791296569817, 'B12': 2555.6047187792947}
STD= {'B1': 474.6178578322387, 'B2': 502.2486875581046, 'B3': 495.5591714117821, 'B4': 572.5308157972637, 'B5': 502.96396851611036, 'B6': 473.0645083903346, 'B7': 512.0212853560603, 'B8': 455.064306633896, 'B8A': 533.614104300756, 'B9': 84.60504420629813, 'B10': 1.7434936778175392, 'B11': 698.0027070808977, 'B12': 668.979590940543}

class NumpyDataset(Dataset):
    def __init__(self, input_dir, label_dir, transform=None):
        self.input_files = sorted(os.listdir(input_dir))
        self.label_files = sorted(os.listdir(label_dir))
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.transform = transform
        # self.label_mapping = {0: -1, 1: 15, 2: 16, 3: 17, 4: 18, 5: 19, 6: 20, 7: 21, 8: 0, 9: 1, 10: 2,11:3,12:4,13:5,14:6,15:7,16:8,17:9,18:10,19:11,20:12,21:13,22:14}
        self.label_mapping = {0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: 0, 9: 1, 10: 2,11:3,12:4,13:5,14:6,15:7,16:8,17:9,18:10,19:11,20:12,21:13,22:-1}

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        input_data = np.load(input_path).astype(np.float32)
        label_data = np.load(label_path)

        if self.transform:
            input_data = torch.from_numpy(input_data)
      
            input_data = self.transform(input_data)
        label_data = np.vectorize(lambda x: self.label_mapping.get(x, -1))(label_data)
        
        return input_data, label_data[0]
def calculate_num_classes(label_dir):
    all_labels = []
    for file in os.listdir(label_dir):
        label_data = np.load(os.path.join(label_dir, file)).astype(np.float32)
        all_labels.append(label_data)
    
    all_labels = np.concatenate(all_labels, axis=None)  # Concaténer tous les labels
    unique_labels = np.unique(all_labels)
    num_classes = len([label for label in unique_labels if label in full_dataset.label_mapping])  # Ne compter que les classes valides

    return [label for label in unique_labels if label in full_dataset.label_mapping],num_classes

# Calculer la moyenne et l'ecart-type pour chaque canal
def calculate_mean_std(input_dir):
    all_data = []
    for file in os.listdir(input_dir):
        data = np.load(os.path.join(input_dir, file))
        all_data.append(data)
    
    all_data = np.concatenate(all_data, axis=0)  # Concaténer sur la dimension des exemples
    mean = np.mean(all_data, axis=(0, 1, 2))  # Moyenne sur les dimensions spatiales et des exemples
    std = np.std(all_data, axis=(0, 1, 2))    # Ecart-type sur les dimensions spatiales et des exemples
    
    return mean, std

# Dossiers des données
input_dir = '/Users/mac/Desktop/docko/tolbi-next-gen/preprocessing-geospatial-data/data/cartagraphie_des_cultures_dataset_2024-07-01_2024-12-01_datasetS2/S2/chips'
label_dir = '/Users/mac/Desktop/docko/tolbi-next-gen/preprocessing-geospatial-data/data/cartagraphie_des_cultures_dataset_2024-07-01_2024-12-01_datasetS2/labels'

# Calcul de la moyenne et de l'écart-type
# mean, std = calculate_mean_std(input_dir)
# print(f"Mean: {mean}, Std: {std}")

# Transformation pour normaliser les données

transform=v2.Compose(
            [
                v2.Normalize(mean=list(MEANS.values()), std=list(STD.values())),
            ]
            ,)


# Créer le dataset
full_dataset = NumpyDataset(input_dir=input_dir, label_dir=label_dir, transform=transform)

# Diviser en train et validation (80% train, 20% validation)
train_size = int(0.80 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Créer les DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# Définir un modèle de segmentation simple (UNet simplifié)
class SimpleUNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(13, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_classes, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
class PretrainedUNet(nn.Module):
    def __init__(self, num_classes):
        super(PretrainedUNet, self).__init__()
        # Charger le modèle ResNet pré-entraîné
        self.encoder = models.resnet50(pretrained=False)
        checkpoint = torch.load("/Users/mac/Desktop/docko/tolbi-next-gen/SSL4EO-S12/data/B13_rn50_dino_0099_ckpt.pth", map_location=torch.device('cpu'))
        self.encoder.load_state_dict(checkpoint, strict=False)
        self.encoder.conv1 = torch.nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder.fc = nn.Identity()
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2]) 
     

        # Décodeur pour ajuster la sortie de l'encodeur (2048, 2, 2) à la résolution souhaitée
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)  # Sortie : (B, 2048, 2, 2)

        x = x.view(x.size(0), 2048, 2, 2)  # Assurer que la sortie est de la bonne forme pour le décodeur
        x = self.decoder(x)  # Sortie : (B, num_classes, 64, 64)
        return x
# Initialiser le modèle, la fonction de perte et l'optimiseur
num_classes=len(set(full_dataset.label_mapping.values()))-1
#model=SimpleUNet(num_classes)
model=PretrainedUNet(num_classes)
# for param in model.encoder.parameters():
#     param.requires_grad = False
criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Utiliser une valeur qui ne correspond à aucune classe pour ignorer
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Boucle d'entraînement
num_epochs = 4
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    for inputs, labels in train_loader:
        labels = labels.long()  # Réduire la dimension des labels et convertir en long
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calcul de la précision
        _, preds = torch.max(outputs, dim=1)
        mask = labels != -1  # Ignorer les pixels de fond
        correct_pixels += torch.sum((preds == labels) & mask).item()
        total_pixels += torch.sum(mask).item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

# Évaluer le modèle sur l'ensemble de validation
import matplotlib.pyplot as plt
model.eval()
val_loss = 0.0
correct_pixels = 0
total_pixels = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        for i in range(min(2, inputs.size(0))):  # Afficher jusqu'à 2 exemples par batch
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title('Label')
            label=labels[i].cpu().numpy()
            mask = label != -1  # Ignorer les pixels de fond

            plt.imshow(label, cmap='viridis')
        

            plt.subplot(1, 2, 2)
            plt.title('Prediction')
            plt.imshow(preds[i].cpu().numpy(), cmap='viridis')

            plt.show()
        labels = labels.long()  # Réduire la dimension des labels et convertir en long
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        # Calcul de la précision
        _, preds = torch.max(outputs, dim=1)
        mask = labels != -1  # Ignorer les pixels de fond
        correct_pixels += torch.sum((preds == labels) & mask).item()
        total_pixels += torch.sum(mask).item()

val_loss = val_loss / len(val_loader)
val_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")