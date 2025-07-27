import os
import requests
import tarfile
import shutil

# Define paths
data_dir = "data"
dataset_url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629951468/mvtec_anomaly_detection.tar.xz"
dataset_file = os.path.join(data_dir, "mvtec_anomaly_detection.tar.xz")
extracted_dir = os.path.join(data_dir, "mvtec_anomaly_detection")

# Create data directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

# Download the dataset
print("Downloading MVTec AD Dataset...")
response = requests.get(dataset_url, stream=True)
if response.status_code == 200:
    with open(dataset_file, 'wb') as f:
        f.write(response.content)
    print("Download complete!")
else:
    print("Failed to download dataset. Please download manually from https://www.mvtec.com/company/research/datasets/mvtec-ad")

# Extract the dataset
print("Extracting dataset...")
with tarfile.open(dataset_file, "r:xz") as tar:
    tar.extractall(data_dir)
print("Extraction complete!")

# Clean up the tar file (optional)
os.remove(dataset_file)