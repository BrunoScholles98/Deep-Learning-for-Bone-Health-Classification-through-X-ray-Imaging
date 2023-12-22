import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import os
import re
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import cv2
from cv2_rolling_ball import subtract_background_rolling_ball

# Definir constantes e caminhos
IMG_PATH = '/d01/scholles/gigasistemica/datasets/CVAT_train/augmented/AUG_NEW_RB_CVAT_Train_FULL_IMG_C1_C2C3/test/C2C3/OPHUB2017-0046.jpg'
MODEL_PATH = '/d01/scholles/gigasistemica/saved_models/Full_Img/efficientnet-b7_AUG_NEW_RB_CVAT_Train_FULL_IMG_C1_C3_Batch4_200Ep/efficientnet-b7_AUG_NEW_RB_CVAT_Train_FULL_IMG_C1_C3_Batch4_200Ep.pth'
MODEL = 'efficientnet-' + re.findall(r'efficientnet-(b\d)', MODEL_PATH)[0] if re.findall(r'efficientnet-(b\d)', MODEL_PATH) else None
PERSONALIZED_RESIZE = True
PERS_RESIZE_NUM = 3
TEST_IMGS_NUMBER = 20

# Definir as dimensões de redimensionamento da imagem de entrada
if PERSONALIZED_RESIZE == True:
    RESIZE = (449, 954)
    print(RESIZE)
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
    diagnosticos = {0: 'Osso Saudável', 1: 'Osso Doente'}
else:
    diagnosticos = {0: 'Alta porosidade óssea não detectada', 1: 'Alta porosidade óssea detectada'}

# Carregar o modelo EfficientNet pré-treinado
device = torch.device('cpu')
model = EfficientNet.from_pretrained(MODEL)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)

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

def saliency(img, model):
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
        
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(np.transpose(input_img.detach().numpy(), (1, 2, 0)))
    plt.xticks([])
    plt.yticks([])
    plt.title('Original Image', fontweight='bold')
    
    plt.subplot(1, 3, 2)
    plt.imshow(slc.numpy(), cmap=plt.cm.hot)
    plt.xticks([])
    plt.yticks([])
    plt.title('Saliency Map', fontweight='bold')
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.xticks([])
    plt.yticks([])
    plt.title('Overlay', fontweight='bold')
    
    diagnosis = 'Diagnóstico: ' + diagnosticos[indices.item()]
    plt.suptitle(diagnosis, fontsize=40, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def rolling_ball(image_path, radius=180):
    image = cv2.imread(image_path, 0)
    final_img, _ = subtract_background_rolling_ball(image, radius, light_background=False,
                                                    use_paraboloid=True, do_presmooth=True)    
    return final_img

def selecionar_imagem():
    file_path = filedialog.askopenfilename(filetypes=[("Imagens", "*.png;*.jpg;*.jpeg")])
    if file_path:
        rmvd_background_img = rolling_ball(file_path)
        rmvd_background_img = Image.fromarray(rmvd_background_img.astype('uint8'), mode='L').convert('RGB')
        saliency(rmvd_background_img, model)


if __name__=="__main__": 
    rmvd_background_img = rolling_ball(IMG_PATH)
    rmvd_background_img = Image.fromarray(rmvd_background_img.astype('uint8'), mode='L').convert('RGB')
    saliency(rmvd_background_img, model)