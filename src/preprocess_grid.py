# src/preprocess_grid.py
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

# Define paths
data_dir = Path("data/grid")
output_dir = Path("data/processed_grid")
train_dir = data_dir / "train"
test_dir = data_dir / "test"
gt_dir = data_dir / "ground_truth"

# Create output directories
os.makedirs(output_dir / "train" / "good", exist_ok=True)
os.makedirs(output_dir / "test" / "good", exist_ok=True)
for defect_type in ["bent", "broken", "glue", "metal_contamination", "thread"]:
    os.makedirs(output_dir / "test" / defect_type, exist_ok=True)
    os.makedirs(output_dir / "ground_truth" / defect_type, exist_ok=True)

print("Preprocessing setup complete. Output directory:", output_dir)

# src/preprocess_grid.py (continued)
def preprocess_image(image_path, target_size=(224, 224), is_mask=False):
    """Preprocess an image or mask: resize and normalize."""
    # Read image
    if is_mask:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE) 
    else:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Resize
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize (ImageNet statistics for RGB images, masks remain 0-255)
    if not is_mask:
        mean = np.array([0.485, 0.456, 0.406]) # ImageNet mean
        std = np.array([0.229, 0.224, 0.225]) # ImageNet std
        image = image / 255.0  # Scale to [0, 1] 
        image = (image - mean) / std # Normalize to ImageNet stats
    
    return image.astype(np.float32) if not is_mask else image.astype(np.uint8)

# Example usage (to be used in the next step)
if __name__ == "__main__":
    sample_image = preprocess_image(data_dir / "train" / "good" / "000.png")
    print("Sample image shape after preprocessing:", sample_image.shape)
    print("Sample image dtype:", sample_image.dtype)

# src/preprocess_grid.py (continued)
def process_train_images():
    """Process all normal training images."""
    good_dir = train_dir / "good"
    output_good_dir = output_dir / "train" / "good"
    
    for image_path in tqdm(list(good_dir.glob("*.png")), desc="Processing train images"):
        # Preprocess image
        processed_image = preprocess_image(image_path)
        
        # Save preprocessed image
        output_path = output_good_dir / image_path.name
        cv2.imwrite(str(output_path), cv2.cvtColor((processed_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    process_train_images()
    print("Training images processed and saved.")

# src/preprocess_grid.py (continued)
def process_test_images_and_masks():
    """Process test images and ground truth masks."""
    defect_types = ["bent", "broken", "glue", "good"]
    
    for defect_type in tqdm(defect_types, desc="Processing test data"):
        test_input_dir = test_dir / defect_type
        output_test_dir = output_dir / "test" / defect_type
        output_gt_dir = output_dir / "ground_truth" / defect_type
        
        for image_path in test_input_dir.glob("*.png"):
            # Preprocess image
            processed_image = preprocess_image(image_path)
            output_image_path = output_test_dir / image_path.name
            cv2.imwrite(str(output_image_path), cv2.cvtColor((processed_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            
            # Process ground truth mask if it exists
            gt_path = gt_dir / defect_type / image_path.name.replace(".png", "_mask.png")
            if gt_path.exists():
                processed_mask = preprocess_image(gt_path, is_mask=True)
                output_mask_path = output_gt_dir / gt_path.name
                cv2.imwrite(str(output_mask_path), processed_mask)

if __name__ == "__main__":
    process_train_images()
    process_test_images_and_masks()
    print("Test images and masks processed and saved.")

# src/preprocess_grid.py (continued)
def summarize_processed_data():
    """Summarize the preprocessed dataset."""
    summary = {
        "train_good": len(list((output_dir / "train" / "good").glob("*.png"))),
        "test_total": sum(len(list((output_dir / "test" / dt).glob("*.png"))) for dt in ["good", "bent", "broken", "glue"]),
        "gt_total": sum(len(list((output_dir / "ground_truth" / dt).glob("*.png"))) for dt in ["bent", "broken", "glue"])
    }
    print("Processed Data Summary:")
    for key, value in summary.items():
        print(f"{key}: {value} images")
    
    # Save summary to file
    with open("data/processed_grid_summary.txt", "w") as f:
        f.write(str(summary))

if __name__ == "__main__":
    process_train_images()
    process_test_images_and_masks()
    summarize_processed_data()
    print("Preprocessing complete.")