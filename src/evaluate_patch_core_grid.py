# src/evaluate_patch_core.py
import os
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import cv2
import matplotlib.pyplot as plt

# Define paths
model_path = Path("models/patch_core_model.pth")
data_dir = Path("data/processed_grid/test")
gt_dir = Path("data/processed_grid/ground_truth")
output_dir = Path("output")
os.makedirs(output_dir, exist_ok=True)

# Set device to CPU (no GPU available)
device = torch.device("cpu")
print("Using device:", device)

# Hyperparameters
input_size = (224, 224)
batch_size = 8  # Adjusted for CPU memory constraints

# src/evaluate_patch_core.py (continued)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class MVTecTestDataset(Dataset):
    def __init__(self, data_dir, gt_dir):
        self.image_paths = []
        self.gt_paths = []
        defect_types = ["good", "bent", "broken", "glue", "metal_contamination", "thread"]
        for dt in defect_types:
            image_paths = list((data_dir / dt).glob("*.png"))
            gt_paths = [gt_dir / dt / p.name.replace(".png", "_mask.png") if (gt_dir / dt / p.name.replace(".png", "_mask.png")).exists() else None for p in image_paths]
            self.image_paths.extend(image_paths)
            self.gt_paths.extend(gt_paths)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        gt_path = self.gt_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, input_size, interpolation=cv2.INTER_AREA)
        image = self.transform(image)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) if gt_path else np.zeros((224, 224), dtype=np.uint8)
        gt = cv2.resize(gt, input_size, interpolation=cv2.INTER_NEAREST)
        return image, gt, image_path.name

# Load dataset and model
dataset = MVTecTestDataset(data_dir, gt_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Load model
checkpoint = torch.load(model_path)
backbone = models.wide_resnet50_2()
backbone = nn.Sequential(*list(backbone.children())[:-1])
backbone.load_state_dict(checkpoint["backbone_state"])
backbone = backbone.to(device)
backbone.eval()
coreset = torch.from_numpy(checkpoint["coreset"]).to(device)

print(f"Loaded {len(dataset)} test images.")


# src/evaluate_patch_core.py (continued, replacing the previous version)
def compute_anomaly_scores(dataloader, backbone, coreset):
    """Compute anomaly scores for test images."""
    scores = []
    filenames = []
    with torch.no_grad():
        # Get feature map size from the first batch
        for batch_images, _, batch_filenames in iter(dataloader):
            batch_images = batch_images.to(device)
            feature_maps = backbone(batch_images)
            _, channels, height, width = feature_maps.shape
            break
        
        for batch_images, _, batch_filenames in tqdm(dataloader, desc="Computing anomaly scores"):
            batch_images = batch_images.to(device)
            feature_maps = backbone(batch_images)
            batch_patches = feature_maps.reshape(batch_images.shape[0] * height * width, channels)
            distances = torch.cdist(batch_patches, coreset)
            patch_scores = torch.min(distances, dim=1)[0]
            image_scores = patch_scores.reshape(batch_images.shape[0], height, width).mean(dim=(1, 2))
            scores.extend(image_scores.cpu().numpy())
            filenames.extend(batch_filenames)
    return scores, filenames

# Compute scores
anomaly_scores, filenames = compute_anomaly_scores(dataloader, backbone, coreset)
print("Anomaly scores computed for", len(anomaly_scores), "images.")