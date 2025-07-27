# Exploring grid dataset structure:
# train/: 1 subdirectories, 264 image files
#   - good: 264 images
# test/: 6 subdirectories, 78 image files
#   - bent: 12 images
#   - broken: 12 images
#   - glue: 11 images
#   - good: 21 images
#   - metal_contamination: 11 images
#   - thread: 11 images
# ground_truth/: 5 subdirectories, 57 image files
#   - bent: 12 images
#   - broken: 12 images
#   - glue: 11 images
#   - metal_contamination: 11 images
#   - thread: 11 images


import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Define paths for grid category
grid_path = Path("data/grid")
normal_image_path = grid_path / "train" / "good" / "000.png"
defective_image_path = grid_path / "test" / "broken" / "000.png"
gt_mask_path = grid_path / "ground_truth" / "broken" / "000_mask.png"

# Load images
normal_image = cv2.imread(str(normal_image_path))
defective_image = cv2.imread(str(defective_image_path))
gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)

# Convert BGR to RGB for display
normal_image_rgb = cv2.cvtColor(normal_image, cv2.COLOR_BGR2RGB)
defective_image_rgb = cv2.cvtColor(defective_image, cv2.COLOR_BGR2RGB)

# Visualize
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Normal Grid")
plt.imshow(normal_image_rgb)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Defective Grid (Broken)")
plt.imshow(defective_image_rgb)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Ground Truth Mask")
plt.imshow(gt_mask, cmap="gray")
plt.axis("off")

plt.show()

# from pathlib import Path

# # Define dataset path
# grid_path = Path("data/grid")

# # Check structure
# print("Exploring grid dataset structure:")
# for folder in ["train", "test", "ground_truth"]:
#     folder_path = grid_path / folder
#     if folder_path.exists():
#         subdirs = [d.name for d in folder_path.iterdir() if d.is_dir()]
#         files = list(folder_path.glob("**/*.png"))
#         print(f"{folder}/: {len(subdirs)} subdirectories, {len(files)} image files")
#         for subdir in subdirs:
#             images = list((folder_path / subdir).glob("*.png"))
#             print(f"  - {subdir}: {len(images)} images")

