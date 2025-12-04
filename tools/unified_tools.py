# Global variable - Choose which tool to execute
TOOL_TO_RUN = 'crop_from_coco'  # Options: 'crop_from_coco', 'rolling_ball_folder', 'rolling_ball_one_img', 'stats_retest'

# Imports
import os
import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Conditional imports (only when needed)
try:
    from cv2_rolling_ball import subtract_background_rolling_ball
except ImportError:
    print("Warning: cv2_rolling_ball not installed. Rolling ball tools will not work.")

try:
    import pyclesperanto_prototype as cle
except ImportError:
    print("Warning: pyclesperanto_prototype not installed. Some rolling ball features may not work.")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchmetrics import Accuracy
    from efficientnet_pytorch import EfficientNet
    import torchvision.transforms as transforms
    from torchvision.datasets import ImageFolder
    from sklearn.metrics import classification_report
except ImportError:
    print("Warning: PyTorch/ML libraries not installed. Stats retest tool will not work.")


# Tool 1: Crop from COCO
def get_crop_dimensions(image_path):
    """Calculate crop dimensions based on image size"""
    # Default values for calculation
    default_n_pixels = 3859324
    default_crop_n_pixels = 360000
    
    image = Image.open(image_path)
    width, height = image.size
    img_n_pixels = width * height
        
    # Calculate crop pixel count
    n_pixels_crop = (default_crop_n_pixels * img_n_pixels) / default_n_pixels
    crop_size = int(np.sqrt(n_pixels_crop))
    
    return crop_size


def crop_from_coco():
    """Crop images from COCO annotations"""
    print("\n" + "="*60)
    print("EXECUTING: Crop from COCO")
    print("="*60 + "\n")
    
    # Configuration
    coco_json_path = "/d01/scholles/gigasistemica/datasets/DXA_HUB/RAW_NOTATIONS/DXA_HUB_C3/annotations/instances_default.json"
    images_folder = "/d01/scholles/gigasistemica/datasets/DXA_HUB/RAW_NOTATIONS/DXA_HUB_C3/images"
    save_folder = "/d01/scholles/gigasistemica/datasets/DXA_HUB/RAW_CROPPED_PR"
    crop_size = 600
    
    # Load COCO JSON file
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)
    
    # Iterate over dataset images with progress bar
    for img_info in tqdm(coco_data["images"], desc="Cropping images"):
        img_id = img_info["id"]
        img_filename = img_info["file_name"]
        img_path = os.path.join(images_folder, img_filename)
    
        img = cv2.imread(img_path)
    
        # Get bounding boxes for this image
        bboxes = [bbox["bbox"] for bbox in coco_data["annotations"] if bbox["image_id"] == img_id]
        
        # Iterate over bounding boxes and crop images
        for bbox in bboxes:
            x, y, w, h = bbox
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
    
            # Check if filename contains "OPRAD"
            if "OPRAD" in img_filename:
                current_crop_size = get_crop_dimensions(img_path)
            else:
                current_crop_size = crop_size
    
            # Calculate crop coordinates
            x_crop = max(0, center_x - current_crop_size // 2)
            y_crop = max(0, center_y - current_crop_size // 2)
            w_crop = current_crop_size
            h_crop = current_crop_size
    
            img_cropped = img[y_crop:y_crop + h_crop, x_crop:x_crop + w_crop]
    
            # Resize to target crop size
            img_cropped = cv2.resize(img_cropped, (crop_size, crop_size))
    
            # Save cropped image
            save_name = f"{img_filename.replace('.jpg', '')}_bbox_{x}_{y}_{w}_{h}.jpg"
            save_path = os.path.join(save_folder, save_name)
            cv2.imwrite(save_path, img_cropped)
    
    print("\n✓ Image cropping process completed.")


# Tool 2: Rolling Ball Folder
def transform_images_in_folder(input_folder, output_folder, radius=180):
    """Apply rolling ball algorithm to all images in a folder"""
    # Walk through all files and folders in input directory
    for root, dirs, files in os.walk(input_folder):
        # Create subfolder structure in output directory
        relative_path = os.path.relpath(root, input_folder)
        output_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(output_subfolder, exist_ok=True)

        # Process each image file in current folder
        for filename in tqdm(files, desc=f"Processing {relative_path}"):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                input_image_path = os.path.join(root, filename)
                output_image_path = os.path.join(output_subfolder, filename)

                image = cv2.imread(input_image_path, 0)

                # Background subtraction using rolling ball
                final_img, background = subtract_background_rolling_ball(image, radius, light_background=False,
                                                                         use_paraboloid=True, do_presmooth=True)

                # Save processed image
                cv2.imwrite(output_image_path, final_img)


def rolling_ball_folder():
    """Process entire folder with rolling ball algorithm"""
    print("\n" + "="*60)
    print("EXECUTING: Rolling Ball - Folder")
    print("="*60 + "\n")
    
    # Configuration
    input_folder = "/mnt/ssd/brunoscholles/GigaSistemica/Datasets/DXA_Osteo_Images"
    output_folder = "/mnt/ssd/brunoscholles/GigaSistemica/Datasets/RB_DXA_Osteo_Images"
    radius = 180
    
    transform_images_in_folder(input_folder, output_folder, radius)
    print("\n✓ Folder processing completed.")


# Tool 3: Rolling Ball One Image
def rolling_ball_one_img():
    """Process single image with rolling ball algorithm"""
    print("\n" + "="*60)
    print("EXECUTING: Rolling Ball - Single Image")
    print("="*60 + "\n")
    
    # Configuration
    output_path = '/d01/scholles/gigasistemica/gigasistemica_sandbox_scholles/tools/image_folder_test_cv/'
    image_path = "/d01/scholles/gigasistemica/datasets/CVAT_train/CVAT_Train_Saudavel_Grave_Croped_600x600/train/Grave/img_OPHUB2018-146.jpg_bbox_[1733.27, 1109.41, 485.17, 58.4].jpg"
    radius_x = 400
    radius_y = 300
    
    image = cv2.imread(image_path, 0)
    cv2.imwrite(output_path + 'original_image.jpg', image)
    
    print('Performing Rolling Ball...')
    final_img = cle.top_hat_sphere(image, radius_x=radius_x, radius_y=radius_y)
    
    # Convert final_img to NumPy array
    final_img = cle.pull(final_img)
    
    cv2.imwrite(output_path + 'cv_output_image.jpg', final_img)
    print(f"\n✓ Image processed and saved to {output_path}")


# Tool 4: Stats Retest
def test_model(model, test_loader, device):
    """Test model and generate statistics"""
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            y_true += labels.cpu().numpy().tolist()
            y_pred += predicted.cpu().numpy().tolist()
    
    report = classification_report(y_true, y_pred, digits=4)
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(report)
    print("✓ Stats saved successfully!")


def stats_retest():
    """Run model testing and generate statistics"""
    print("\n" + "="*60)
    print("EXECUTING: Stats Retest")
    print("="*60 + "\n")
    
    # Configuration
    MODEL_PATH = '/d01/scholles/gigasistemica/saved_models/old/efficientnet-b7_AUG_RB_CVAT_Train_C1_C2C3_Croped_600x600_Batch4_100Ep/efficientnet-b7_AUG_RB_CVAT_Train_Saudavel_DoenteGeral_Croped_600x600_Batch4_100Ep.pth'
    MODEL = 'efficientnet-b7'
    DATASET_PATH = Path('/d01/scholles/gigasistemica/datasets/CVAT_train/roll_ball_only/RB_NEW_CVAT_C1_C2C3_Cropped_600x600_copy')
    BATCH_SIZE = 4
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Determine image size based on model
    if MODEL == 'efficientnet-b0':
        RESIZED = (224, 224)
    elif MODEL == 'efficientnet-b3':
        RESIZED = (300, 300)
    elif MODEL == 'efficientnet-b4':
        RESIZED = (380, 380)
    elif MODEL == 'efficientnet-b6':
        RESIZED = (528, 528)
    elif MODEL == 'efficientnet-b7':
        RESIZED = (600, 600)
    else:
        RESIZED = (224, 224)  # default

    print(f"Loading model: {MODEL}")
    print(f"Device: {DEVICE}")
    
    model = EfficientNet.from_pretrained(MODEL)
    state_dict = torch.load(MODEL_PATH)
    model.load_state_dict(state_dict)
    model.to(DEVICE)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    
    transform = transforms.Compose([
        transforms.Resize(RESIZED),
        transforms.ToTensor(),
        normalize
    ])

    print(f"Loading dataset from: {DATASET_PATH / 'test'}")
    test_dataset = ImageFolder(DATASET_PATH / "test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_model(model, test_loader, DEVICE)


# Main execution
def main():
    """Main execution function - routes to selected tool"""
    print("\n" + "="*60)
    print("UNIFIED TOOLS - Image Processing & Model Testing")
    print("="*60)
    print(f"Selected Tool: {TOOL_TO_RUN}")
    
    if TOOL_TO_RUN == 'crop_from_coco':
        crop_from_coco()
    
    elif TOOL_TO_RUN == 'rolling_ball_folder':
        rolling_ball_folder()
    
    elif TOOL_TO_RUN == 'rolling_ball_one_img':
        rolling_ball_one_img()
    
    elif TOOL_TO_RUN == 'stats_retest':
        stats_retest()
    
    else:
        print(f"\n❌ ERROR: Unknown tool '{TOOL_TO_RUN}'")
        print("\nAvailable options:")
        print("  - 'crop_from_coco'")
        print("  - 'rolling_ball_folder'")
        print("  - 'rolling_ball_one_img'")
        print("  - 'stats_retest'")
        return
    
    print("\n" + "="*60)
    print("EXECUTION COMPLETED")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
