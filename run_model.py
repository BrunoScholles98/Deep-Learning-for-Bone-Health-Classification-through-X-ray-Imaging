import matplotlib.pyplot as plt
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import cv2
from cv2_rolling_ball import subtract_background_rolling_ball
import json
import os
from datetime import datetime
import sys

MODEL_PATH = sys.argv[1]
#MODEL_PATH =  '/d01/scholles/gigasistemica/gigacandanga_exec/model/efficientnet-b7_AUG_NEW_RB_CVAT_Train_FULL_IMG_C1_C3_Batch4_200Ep.pth'
MODEL = 'efficientnet-' + re.findall(r'efficientnet-(b\d)', MODEL_PATH)[0] if re.findall(r'efficientnet-(b\d)', MODEL_PATH) else None
PERSONALIZED_RESIZE = True

# Definir as dimensões de redimensionamento da imagem de entrada
if PERSONALIZED_RESIZE == True:
    RESIZE = (449, 954)
else:
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
    RESIZE = model_resize_map.get(MODEL, None)

if RESIZE is None:
    raise ValueError("Tamanho de redimensionamento não definido para o modelo selecionado.")

if 'C2C3' in MODEL_PATH:
    diagnosticos = {0: 'Healthy Bone', 1: 'Diseased Bone'}
else:
    diagnosticos = {0: 'Healthy Patient', 1: 'Patient with Osteoporosis'}

# Carregar o modelo EfficientNet pré-treinado
device = torch.device('cpu')
model = EfficientNet.from_pretrained(MODEL, MODEL_PATH)
#state_dict = torch.load(MODEL_PATH, map_location=device)
#model.load_state_dict(state_dict)

# Definir as transformações para pré-processar a imagem de entrada no formato esperado pelo modelo
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)

transform = transforms.Compose([
    transforms.Resize(RESIZE),
    transforms.ToTensor(),
    normalize
])

def save_image_with_metadata(image, img_path, image_type, output_path):
    # Extrair o nome original da imagem do caminho
    img_name = os.path.basename(img_path)
    img_name, _ = os.path.splitext(img_name)

    # Adicionar data e horário ao nome do arquivo
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"{img_name}_{image_type}_{current_time}.png"
    file_path = os.path.join(output_path, file_name)

    Image.fromarray((image * 255).astype(np.uint8)).save(file_path, "PNG")
    return file_path

def saliency(img, model, img_path, output_path):
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    input = transform(img)
    input.unsqueeze_(0)
    input.requires_grad = True
    preds = model(input)
    score, indices = torch.max(preds, 1)
    score.backward()
    slc, _ = torch.max(torch.abs(input.grad[0]), dim=0)
    slc = (slc - slc.min())/(slc.max()-slc.min())

    with torch.no_grad():
        input_img = inv_normalize(input[0])

    imagem_hot = plt.cm.hot(slc)
    original = np.clip(np.transpose(input_img.detach().numpy(), (1, 2, 0)), 0, 1)
    intensity = 1.5
    red_channel = imagem_hot[:, :, 0] * intensity
    threshold = 0.3
    red_channel = np.where(red_channel < threshold, 0, red_channel)
    overlay = np.clip(original + red_channel[:, :, np.newaxis], 0, 1)

    imagem_hot_path = save_image_with_metadata(imagem_hot, img_path, "saliency", output_path)
    overlay_path = save_image_with_metadata(overlay, img_path, "overlay", output_path)

    diagnosis = diagnosticos[indices.item()]
    
    return diagnosis, imagem_hot_path, overlay_path

def rolling_ball(image_path, radius=180):
    image = cv2.imread(image_path, 0)
    removed_bg_img, _ = subtract_background_rolling_ball(image, radius, light_background=False,
                                                    use_paraboloid=True, do_presmooth=True)    
    return removed_bg_img

def read_json(json_string):
    try:
        json_data = json.loads(json_string)
        img_path = json_data["img_path"]
        destination_path = json_data["destination_path"]

    except json.JSONDecodeError as e:
        print("Erro ao analisar o JSON:", e)
    except KeyError as e:
        print("Chave ausente no JSON:", e)
        
    return img_path, destination_path

if __name__=="__main__":
    while True:
        json_string = input()
        try:
            img_path, destination_path = read_json(json_string)        
            rmvd_background_img = rolling_ball(img_path)
            rmvd_background_img = Image.fromarray(rmvd_background_img.astype('uint8'), mode='L').convert('RGB')
            diagnosis, imagem_hot_path, overlay_path = saliency(rmvd_background_img, model, img_path, destination_path)
            
            # Generate JSON output
            json_output = {
                "result": "ok",
                "diagnosis": diagnosis,
                "saliency_img_path": imagem_hot_path,
                "overlay_img_path": overlay_path
            }

            # Convert the dictionary to a JSON string
            json_output_string = json.dumps(json_output)

            # Print or do something else with the JSON string
            print(json_output_string)
        
        except:
            json_output = {
                "result": "fail",
                "diagnosis": "",
                "saliency_img_path": "",
                "overlay_img_path": ""
            }
            
            json_output_string = json.dumps(json_output)
            print(json_output_string)