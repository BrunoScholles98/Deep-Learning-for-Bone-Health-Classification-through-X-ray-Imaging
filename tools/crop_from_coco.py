import cv2
import json
import os
from tqdm import tqdm
from PIL import Image
import numpy as np

def obter_dimensoes_recorte(caminho_imagem):
    # Valores padrão para o cálculo
    default_n_pixels = 3859324
    default_crop_n_pixels = 360000
    
    # Abre a imagem usando a biblioteca PIL
    imagem = Image.open(caminho_imagem)
    
    # Obtém as dimensões da imagem (largura x altura)
    largura, altura = imagem.size
    
    # Calcula o número total de pixels na imagem
    img_n_pixels = largura * altura
        
    # Calcula o número de pixels para o recorte desejado
    n_pixels_corte = (default_crop_n_pixels * img_n_pixels) / default_n_pixels
    
    # Calcula o tamanho do recorte em pixels
    tam_recorte = int(np.sqrt(n_pixels_corte))
    
    return tam_recorte

# Caminho para o arquivo COCO JSON contendo as informações das bounding boxes
coco_json_path = "/d01/scholles/gigasistemica/datasets/DXA_HUB/RAW_NOTATIONS/DXA_HUB_C3/annotations/instances_default.json"

# Caminho para a pasta onde as imagens originais estão armazenadas
imagens_pasta = "/d01/scholles/gigasistemica/datasets/DXA_HUB/RAW_NOTATIONS/DXA_HUB_C3/images"

# Caminho para a pasta onde as imagens recortadas serão salvas
salvar_pasta = "/d01/scholles/gigasistemica/datasets/DXA_HUB/RAW_CROPPED_PR"

# Tamanho desejado para as imagens recortadas
tamanho_recorte = 600

# Carregar o arquivo COCO JSON
with open(coco_json_path, "r") as f:
    coco_data = json.load(f)

# Iterar sobre as imagens do dataset com a barra de progresso
for img_info in tqdm(coco_data["images"], desc="Recortando imagens"):
    img_id = img_info["id"]
    img_filename = img_info["file_name"]
    img_path = os.path.join(imagens_pasta, img_filename)

    # Carregar a imagem
    img = cv2.imread(img_path)

    # Obter as bounding boxes da imagem
    bboxes = [bbox["bbox"] for bbox in coco_data["annotations"] if bbox["image_id"] == img_id]
    
    # Iterar sobre as bounding boxes e recortar as imagens
    for bbox in bboxes:
        x, y, w, h = bbox
        centro_x = int(x + w / 2)
        centro_y = int(y + h / 2)

        # Verifica se o nome do arquivo contém "OPRAD"
        if "OPRAD" in img_filename:
            # Obter o tamanho de recorte personalizado
            tam_recorte = obter_dimensoes_recorte(img_path)
        else:
            # Usar o tamanho de recorte padrão para as demais imagens
            tam_recorte = tamanho_recorte

        # Calcular as coordenadas para o recorte
        x_recorte = max(0, centro_x - tam_recorte // 2)
        y_recorte = max(0, centro_y - tam_recorte // 2)
        w_recorte = tam_recorte
        h_recorte = tam_recorte

        # Recortar a imagem
        img_recortada = img[y_recorte:y_recorte + h_recorte, x_recorte:x_recorte + w_recorte]

        # Redimensionar a imagem para o tamanho desejado (224x224)
        img_recortada = cv2.resize(img_recortada, (tamanho_recorte, tamanho_recorte))

        # Salvar a imagem recortada
        nome_salvar = f"{img_filename.replace('.jpg', '')}_bbox_{x}_{y}_{w}_{h}.jpg"
        caminho_salvar = os.path.join(salvar_pasta, nome_salvar)
        cv2.imwrite(caminho_salvar, img_recortada)

        #print(f"Imagem recortada {nome_salvar} salva.")

print("Conclusão do processo de recorte de imagens.")