import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import v2
MEANS= {'B1': 1943.5765059045868, 'B2': 1770.5042365802472, 'B3': 1841.929756298748, 'B4': 2023.9325451849686, 'B5': 2196.810997358337, 'B6': 3036.156983701043, 'B7': 3550.6760031532895, 'B8': 3257.76563697493, 'B8A': 3878.0640571977488, 'B9': 453.6613090328172, 'B10': 16.34099485241143, 'B11': 3725.222169609389, 'B12': 2631.5447871529555}
STD= {'B1': 338.7211177621409, 'B2': 382.74496679123246, 'B3': 381.36465021332884, 'B4': 516.3111899873122, 'B5': 448.8244821301182, 'B6': 391.06762112698544, 'B7': 428.20975027074667, 'B8': 397.0994118544168, 'B8A': 452.0902742127855, 'B9': 61.61428274428779, 'B10': 1.6713470991052297, 'B11': 805.0769203085229, 'B12': 789.4134654722126}
class NumpyDataset(Dataset):
    def __init__(self, input_dir, label_dir, transform=None):
        self.input_files = sorted(os.listdir(input_dir))
        self.label_files = sorted(os.listdir(label_dir))
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.transform = transform

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

        return input_data, label_data

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
input_dir = '/Users/mac/Desktop/docko/tolbi-next-gen/preprocessing-geospatial-data/data/datasetS2/S2/chips'
label_dir = '/Users/mac/Desktop/docko/tolbi-next-gen/preprocessing-geospatial-data/data/datasetS2/labels'

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
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Créer les DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Exemple d'itération sur le DataLoader
for inputs, labels in train_loader:
    print(f"Inputs batch shape: {inputs.shape}")
    print(f"Labels batch shape: {labels.shape}")
   