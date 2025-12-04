import os
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import cv2
from cv2_rolling_ball import subtract_background_rolling_ball
from tqdm import tqdm

# Paths and settings
MODEL_PATH = '/mnt/ssd/brunoscholles/GigaSistemica/Models/Osteoporosis/efficientnet-b7_FULL_IMG_C1_C3.pth'
INPUT_FOLDER = '/mnt/nas/BrunoScholles/Gigasistemica/Datasets/SLM/ATH_SLM_resized'
OUTPUT_FOLDER = '/mnt/nas/BrunoScholles/Gigasistemica/SLM_Outputs/Sep_Atheroma_Dataset/Osteo'
PERSONALIZED_RESIZE = True
CUSTOM_RESIZE = (449, 954)
model_resize_map = {
    'efficientnet-b0': (224, 224),
    'efficientnet-b1': (240, 240),
    'efficientnet-b2': (260, 260),
    'efficientnet-b3': (300, 300),
    'efficientnet-b4': (380, 380),
    'efficientnet-b5': (456, 456),
    'efficientnet-b6': (528, 528),
    'efficientnet-b7': (600, 600)
}
ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# Load model and set resize
MODEL = None
match = re.findall(r'efficientnet-(b\d)', MODEL_PATH)
if match:
    MODEL = 'efficientnet-' + match[0]

if PERSONALIZED_RESIZE:
    RESIZE = CUSTOM_RESIZE
else:
    if MODEL is None:
        raise ValueError("Could not determine model type from MODEL_PATH for resize.")
    RESIZE = model_resize_map.get(MODEL)
    if not RESIZE:
        raise ValueError("No resize size for selected model.")

labels = {0: 'Healthy Bone', 1: 'Diseased Bone'} if 'C2C3' in MODEL_PATH else {0: 'Healthy Patient', 1: 'Osteoporosis'}

model = EfficientNet.from_pretrained(MODEL, MODEL_PATH)
model.to(device)
for param in model.parameters():
    param.requires_grad = False

# Transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])
transform = transforms.Compose([transforms.Resize(RESIZE), transforms.ToTensor(), normalize])

def save_image_with_metadata(img_array, img_path, tag, flag, out_dir, is_normalized=True):
    base = os.path.splitext(os.path.basename(img_path))[0]
    name = f"{base}_{tag}_{flag}.png"
    path = os.path.join(out_dir, name)
    save_array = img_array
    if is_normalized:
        save_array = (img_array * 255)
    Image.fromarray(save_array.astype(np.uint8)).save(path, "PNG")
    return path


def saliency(img, model, img_path, out_dir):
    model.eval()
    inp = transform(img).unsqueeze(0).to(device)
    inp.requires_grad = True

    preds = model(inp)
    score, idx = torch.max(preds, 1)
    score.backward()

    grad = torch.abs(inp.grad[0]).cpu()
    sal_map, _ = torch.max(grad, 0)
    sal_map = (sal_map - sal_map.min()) / (sal_map.max() - sal_map.min() + 1e-8)

    with torch.no_grad():
        img_orig = inv_normalize(inp[0].cpu())
    orig = np.clip(np.transpose(img_orig.numpy(), (1,2,0)), 0, 1)

    cmap = plt.get_cmap('hot')(sal_map.numpy())
    intensity = 1.5
    red_channel = cmap[:, :, 0] * intensity
    threshold = 0.3
    red_channel = np.where(red_channel < threshold, 0, red_channel)

    # Create a red overlay instead of white
    red_overlay = np.zeros_like(orig)
    red_overlay[:,:,0] = red_channel
    overlay = np.clip(orig + red_overlay, 0, 1)

    flag = "true" if idx.item() == 1 else "false"
    diagnosis = labels[int(idx.item())]

    sal_path = save_image_with_metadata(cmap, img_path, "saliency", flag, out_dir, is_normalized=True)
    ov_path = save_image_with_metadata(overlay, img_path, "overlay", flag, out_dir, is_normalized=True)
    return diagnosis, sal_path, ov_path, flag


def rolling_ball(image_path, radius=180):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    bg_removed, _ = subtract_background_rolling_ball(img, radius, light_background=False, use_paraboloid=True, do_presmooth=True)
    return bg_removed


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    images = [os.path.join(INPUT_FOLDER, f) for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(ALLOWED_EXTENSIONS)]
    if not images:
        print("No images found in input folder.")
        return

    to_run = []
    for path in images:
        base = os.path.splitext(os.path.basename(path))[0]
        true_path = os.path.join(OUTPUT_FOLDER, f"{base}_saliency_true.png")
        false_path = os.path.join(OUTPUT_FOLDER, f"{base}_saliency_false.png")
        if not (os.path.exists(true_path) or os.path.exists(false_path)):
            to_run.append(path)
        else:
            tqdm.write(f"Skipping already processed: {path}")

    if not to_run:
        print("All images processed.")
        return

    for path in tqdm(to_run, desc="Processing images", total=len(to_run)):
        tqdm.write(f"Working on {path}...")
        try:
            bg = rolling_ball(path)
            pil_img = Image.fromarray(bg.astype('uint8'), 'L').convert('RGB')

            diag, sal_path, ov_path, flag = saliency(pil_img, model, path, OUTPUT_FOLDER)

            rb_path = save_image_with_metadata(bg, path, "rolling_ball", flag, OUTPUT_FOLDER, is_normalized=False)

            tqdm.write(f"Result for {os.path.basename(path)}: {diag}")
            tqdm.write(f"Saved rolling ball image: {rb_path}")
            tqdm.write(f"Saved saliency map: {sal_path}")
            tqdm.write(f"Saved overlay: {ov_path}")
            tqdm.write("-"*40)
        except Exception as e:
            tqdm.write(f"Error on {path}: {e}")

if __name__ == "__main__":
    main()