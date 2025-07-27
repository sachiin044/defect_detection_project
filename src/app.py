# src/app.py
import os
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
import streamlit as st
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
from tqdm import tqdm

# Define paths
model_path = Path("models/patch_core_model.pth")
data_dir = Path("data/processed_grid/test")
gt_dir = Path("data/processed_grid/ground_truth")

# Set device to CPU
device = torch.device("cpu")

# Load the trained model
checkpoint = torch.load(model_path)
backbone = models.wide_resnet50_2()
backbone = nn.Sequential(*list(backbone.children())[:-1])
backbone.load_state_dict(checkpoint["backbone_state"])
backbone = backbone.to(device)
backbone.eval()
coreset = torch.from_numpy(checkpoint["coreset"]).to(device)

# Set up the Streamlit app
st.title("Patch Core Anomaly Detection - Grid Category with Drag-and-Drop")

# Define the dataset class for preloaded images
class MVTecTestDataset(Dataset):
    def __init__(self, data_dir, gt_dir):
        self.image_paths = []
        self.gt_paths = []
        defect_types = ["good", "bent", "broken", "glue"]
        for dt in defect_types:
            image_paths = list((data_dir / dt).glob("*.png"))
            gt_paths = [gt_dir / dt / p.name.replace(".png", "_mask.png") if (gt_dir / dt / p.name.replace(".png", "_mask.png")).exists() else None for p in image_paths]
            self.image_paths.extend(image_paths)
            self.gt_paths.extend(gt_paths)
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for scoring
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        gt_path = self.gt_paths[idx]
        # Load and preprocess image for scoring
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image_tensor = self.transform(image)  # Normalized tensor for scoring
        # Load original image for display
        orig_image = cv2.imread(str(image_path))
        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        orig_image = cv2.resize(orig_image, (224, 224), interpolation=cv2.INTER_AREA)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) if gt_path else np.zeros((224, 224), dtype=np.uint8)
        gt = cv2.resize(gt, (224, 224), interpolation=cv2.INTER_NEAREST)
        return image_tensor, orig_image, gt, image_path.name  # Return four values

# Load dataset for preloaded images
dataset = MVTecTestDataset(data_dir, gt_dir)

# Define the dataset class for uploaded images
class UploadDataset(Dataset):
    def __init__(self, image):
        self.image = image
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        img = cv2.imdecode(np.frombuffer(self.image.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        image_tensor = self.transform(img)
        orig_image = img  # Keep original for display
        return image_tensor, orig_image

# Define anomaly score and heatmap computation function
def compute_anomaly_heatmap(image_tensor, backbone, coreset):
    """Compute patch-level anomaly scores and generate a heatmap."""
    with torch.no_grad():
        feature_maps = backbone(image_tensor.unsqueeze(0).to(device))
        _, channels, height, width = feature_maps.shape
        patches = feature_maps.reshape(height * width, channels)
        distances = torch.cdist(patches, coreset)
        patch_scores = torch.min(distances, dim=1)[0].cpu().numpy()
        heatmap = patch_scores.reshape(height, width)
        # Upsample heatmap to match image size
        heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # Normalize to [0, 1]
        return heatmap

def compute_anomaly_score(image_tensor, backbone, coreset):
    """Compute anomaly score for a single normalized image tensor."""
    with torch.no_grad():
        feature_maps = backbone(image_tensor.unsqueeze(0).to(device))
        _, channels, height, width = feature_maps.shape
        patches = feature_maps.reshape(height * width, channels)
        distances = torch.cdist(patches, coreset)
        score = torch.min(distances, dim=1)[0].mean().item()
    return score

# File uploader for drag-and-drop
uploaded_file = st.file_uploader("Upload a defective image", type=["png", "jpg", "jpeg"])

# Handle uploaded image or selected image
if uploaded_file is not None:
    # Use uploaded image
    upload_dataset = UploadDataset(uploaded_file)
    image_tensor, orig_image = upload_dataset[0]  # Unpack two values
    filename = uploaded_file.name
else:
    # Use selected image from dataset
    selected_idx = image_options.index(selected_image) if 'selected_image' in locals() else 0
    image_tensor, orig_image, gt, filename = dataset[selected_idx]

# Compute anomaly score and heatmap
score = compute_anomaly_score(image_tensor, backbone, coreset)
heatmap = compute_anomaly_heatmap(image_tensor, backbone, coreset)

# Overlay heatmap on original image
heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(orig_image, 0.7, heatmap_colored, 0.3, 0)

# Display results
st.header("Anomaly Detection Result")
st.write(f"Image: {filename}")
st.write(f"Anomaly Score: {score:.2f}")
st.image(orig_image, caption="Original Image", channels="RGB", use_container_width=True)
st.image(overlay, caption="Defect Heatmap Overlay", channels="RGB", use_container_width=True)

# Sidebar for image selection (only if no upload)
if uploaded_file is None:
    st.sidebar.header("Select Image")
    image_options = [p.name for p in dataset.image_paths]
    selected_image = st.sidebar.selectbox("Choose an image", image_options)

# Compute all scores for summary (using dataset)
all_scores = []
for image_tensor, orig_image, gt, filename in tqdm(dataset, desc="Computing all scores"):
    score = compute_anomaly_score(image_tensor, backbone, coreset)
    all_scores.append((filename, score))

st.sidebar.header("Summary")
st.sidebar.write("Average Anomaly Score:", np.mean([s[1] for s in all_scores]))

if st.sidebar.button("Save Results"):
    with open(Path("output") / "anomaly_scores.txt", "w") as f:
        for filename, score in all_scores:
            f.write(f"{filename}: {score:.2f}\n")
    st.sidebar.success("Results saved to output/anomaly_scores.txt")