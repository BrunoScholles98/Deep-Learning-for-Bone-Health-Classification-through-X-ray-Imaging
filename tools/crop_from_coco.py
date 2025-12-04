import cv2
import json
import os
from tqdm import tqdm
from PIL import Image
import numpy as np

def obter_dimensoes_recorte(caminho_imagem):
    # Default values for calculation
    default_n_pixels = 3859324
    default_crop_n_pixels = 360000
    
    imagem = Image.open(caminho_imagem)
    largura, altura = imagem.size
    img_n_pixels = largura * altura
        
    # Calculate crop pixel count
    n_pixels_corte = (default_crop_n_pixels * img_n_pixels) / default_n_pixels
    tam_recorte = int(np.sqrt(n_pixels_corte))
    
    return tam_recorte

# Path to COCO JSON file with bounding box annotations
coco_json_path = "/d01/scholles/gigasistemica/datasets/DXA_HUB/RAW_NOTATIONS/DXA_HUB_C3/annotations/instances_default.json"

# Path to folder containing the original images
imagens_pasta = "/d01/scholles/gigasistemica/datasets/DXA_HUB/RAW_NOTATIONS/DXA_HUB_C3/images"

# Path to folder where cropped images will be saved
salvar_pasta = "/d01/scholles/gigasistemica/datasets/DXA_HUB/RAW_CROPPED_PR"

# Desired crop size
tamanho_recorte = 600

# Load COCO JSON file
with open(coco_json_path, "r") as f:
    coco_data = json.load(f)

# Iterate over dataset images with progress bar
for img_info in tqdm(coco_data["images"], desc="Cropping images"):
    img_id = img_info["id"]
    img_filename = img_info["file_name"]
    img_path = os.path.join(imagens_pasta, img_filename)

    img = cv2.imread(img_path)

    # Get bounding boxes for this image
    bboxes = [bbox["bbox"] for bbox in coco_data["annotations"] if bbox["image_id"] == img_id]
    
    # Iterate over bounding boxes and crop images
    for bbox in bboxes:
        x, y, w, h = bbox
        centro_x = int(x + w / 2)
        centro_y = int(y + h / 2)

        # Check if filename contains "OPRAD"
        if "OPRAD" in img_filename:
            tam_recorte = obter_dimensoes_recorte(img_path)
        else:
            tam_recorte = tamanho_recorte

        # Calculate crop coordinates
        x_recorte = max(0, centro_x - tam_recorte // 2)
        y_recorte = max(0, centro_y - tam_recorte // 2)
        w_recorte = tam_recorte
        h_recorte = tam_recorte

        img_recortada = img[y_recorte:y_recorte + h_recorte, x_recorte:x_recorte + w_recorte]

        # Resize to target crop size
        img_recortada = cv2.resize(img_recortada, (tamanho_recorte, tamanho_recorte))

        # Save cropped image
        nome_salvar = f"{img_filename.replace('.jpg', '')}_bbox_{x}_{y}_{w}_{h}.jpg"
        caminho_salvar = os.path.join(salvar_pasta, nome_salvar)
        cv2.imwrite(caminho_salvar, img_recortada)

print("Image cropping process completed.")