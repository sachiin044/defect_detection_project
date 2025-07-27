# src/train_patch_core.py
import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans

# Define paths
data_dir = Path("data/processed_grid/train/good")
output_dir = Path("models")
os.makedirs(output_dir, exist_ok=True)

# Set device to CPU (no GPU available)
device = torch.device("cpu")
print("Using device:", device)

# Hyperparameters
input_size = (224, 224)
batch_size = 8  # Adjusted for CPU memory constraints
num_clusters = 100  # Number of clusters for k-means

# src/train_patch_core.py (continued)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class MVTecDataset(Dataset):
    def __init__(self, data_dir):
        self.image_paths = list(data_dir.glob("*.png"))
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Reapply normalization
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, input_size, interpolation=cv2.INTER_AREA)
        image = self.transform(image)
        return image

# Load dataset
dataset = MVTecDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"Loaded {len(dataset)} training images.")

# src/train_patch_core.py (continued)
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

# Load model with proper weights
weights = Wide_ResNet50_2_Weights.DEFAULT
backbone = wide_resnet50_2(weights=weights)

# Remove final classification layer for feature extraction
backbone = nn.Sequential(*list(backbone.children())[:-1])
backbone = backbone.to(device)
backbone.eval()

def extract_features(dataloader):
    """Extract features from all images using the backbone."""
    features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            batch = batch.to(device)
            feature_maps = backbone(batch)
            features.append(feature_maps.cpu().numpy())
    return np.concatenate(features, axis=0)

# Extract features
features = extract_features(dataloader)
print("Feature shape:", features.shape)  # Expected shape: (num_images, channels, height, width)

# src/train_patch_core.py (continued)
def extract_patches(features):
    """Extract patches from feature maps."""
    batch_size, channels, height, width = features.shape
    patches = features.reshape(batch_size * height * width, channels)
    return patches

def cluster_patches(patches, num_clusters):
    """Cluster patches using k-means."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(patches)
    return kmeans.cluster_centers_, cluster_labels

# Extract patches and cluster
patches = extract_patches(features)
cluster_centers, _ = cluster_patches(patches, num_clusters)
print("Cluster centers shape:", cluster_centers.shape)  # Expected: (100, 2048)

# src/train_patch_core.py (continued)
def select_coreset(cluster_centers, coreset_size=50):
    """Select a coreset from cluster centers."""
    # Simple selection: take top variance centers (can be improved with advanced sampling)
    variances = np.var(cluster_centers, axis=1)
    indices = np.argsort(variances)[::-1][:coreset_size]
    return cluster_centers[indices]

# Select coreset
coreset = select_coreset(cluster_centers)
print("Coreset shape:", coreset.shape)  # Expected: (50, 2048)

# Save model components
torch.save({"backbone_state": backbone.state_dict(), "coreset": coreset}, output_dir / "patch_core_model.pth")
print("Model saved to:", output_dir / "patch_core_model.pth")

