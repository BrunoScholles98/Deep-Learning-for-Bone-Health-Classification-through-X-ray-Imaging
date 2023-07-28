import sys
import torch
import cv2
import numpy as np
import os
from PIL import Image
from torchvision import models
from efficientnet_pytorch import EfficientNet
from gigasistemica_sandbox_scholles.heatmaps.grad_cam import GradCam, GuidedBackpropReLUModel, show_cams, show_gbs, preprocess_image
sys.path.append('/')

MODEL = 'efficientnet-b7'
HEATMAP_FOLDER = '/d01/scholles/gigasistemica/gigasistemica_sandbox_scholles/heatmaps'
MODEL_PATH = '/d01/scholles/gigasistemica/gigasistemica_sandbox_scholles/results/AUG_RB_Croped_CVAT2_EfficientNetB7_Batch4_Saudavel_Grave_600x600_100Ep/AUG_RB_Croped_CVAT2_EfficientNetB7_Batch4_Saudavel_Grave_600x600_100Ep.pth'
IMG_PATH = '/d01/scholles/gigasistemica/datasets/CVAT_train/AUG_RB_CVAT_Train_Saudavel_Grave_Croped_600x600/val/Grave/OPHUB2022-1 [1828.78, 1185.4, 413.65, 89.23].jpg'
CUDA_BOOL = True
PERSONALIZED_RESIZE = False

if MODEL == 'efficientnet-b0':
    TARGER_LAYERS = ['1', '10', '15']
elif MODEL == 'efficientnet-b3':
    TARGER_LAYERS = ['1', '12', '25']
elif MODEL == 'efficientnet-b4':
    TARGER_LAYERS = ['1', '15', '31']
elif MODEL == 'efficientnet-b6':
    TARGER_LAYERS = ['1', '22', '44']
else:
    TARGER_LAYERS = ['1', '27', '54']

if PERSONALIZED_RESIZE == True:
    RESIZED = ((lambda img: (img.size[0] // 4, img.size[1] // 4))(Image.open(IMG_PATH)))
else:
    if MODEL == 'efficientnet-b0':
        RESIZED = (224,224)
    elif MODEL == 'efficientnet-b3':
        RESIZED = (300,300)
    elif MODEL == 'efficientnet-b4':
        RESIZED = (380,380)
    elif MODEL == 'efficientnet-b6':
        RESIZED = (528,528)
    elif MODEL == 'efficientnet-b7':
        RESIZED = (600,600)

def limpar_pasta(path):
    for arquivo in os.listdir(path):
        path_completo = os.path.join(path, arquivo)
        if os.path.isfile(path_completo):
            os.remove(path_completo)

if __name__ == '__main__':    
    limpar_pasta(HEATMAP_FOLDER)
    
    img = cv2.imread(IMG_PATH, 1)
    cv2.imwrite('/d01/scholles/gigasistemica/gigasistemica_sandbox_scholles/heatmaps/imagem_original.jpg', img)
    
    model = EfficientNet.from_pretrained(MODEL)
    state_dict = torch.load(MODEL_PATH)
    model.load_state_dict(state_dict)

    grad_cam = GradCam(model=model, blob_name='_blocks', target_layer_names=TARGER_LAYERS, use_cuda=CUDA_BOOL, size=RESIZED)
    
    img = np.float32(cv2.resize(img, RESIZED)) / 255
    inputs = preprocess_image(img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    mask_dic = grad_cam(inputs, target_index)
    show_cams(img, mask_dic)

    gb_model = GuidedBackpropReLUModel(model=model, activation_layer_name='MemoryEfficientSwish', use_cuda=CUDA_BOOL)
    show_gbs(inputs, gb_model, target_index, mask_dic)
