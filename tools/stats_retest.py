import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from torchvision.models import efficientnet_b7
from efficientnet_pytorch import EfficientNet

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from PIL import Image
import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_model(model, test_loader):
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            y_true += labels.cpu().numpy().tolist()
            y_pred += predicted.cpu().numpy().tolist()
    
    report = classification_report(y_true, y_pred, digits=4)
    print(report)    

    # Confirme que o arquivo foi salvo
    print("Arquivo salvo com sucesso!")

if __name__ == "__main__":
    MODEL_PATH = '/d01/scholles/gigasistemica/saved_models/old/efficientnet-b7_AUG_RB_CVAT_Train_C1_C2C3_Croped_600x600_Batch4_100Ep/efficientnet-b7_AUG_RB_CVAT_Train_Saudavel_DoenteGeral_Croped_600x600_Batch4_100Ep.pth'
    MODEL = 'efficientnet-b7'
    DATASET_PATH = Path('/d01/scholles/gigasistemica/datasets/CVAT_train/roll_ball_only/RB_NEW_CVAT_C1_C2C3_Cropped_600x600_copy')
    BATCH_SIZE = 4

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

    #load pretrained resnet model
    model = EfficientNet.from_pretrained(MODEL)
    state_dict = torch.load(MODEL_PATH)
    model.load_state_dict(state_dict)
    model.to(DEVICE)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    
    transform = transforms.Compose([ # Define as transformações que serão aplicadas às imagens
        transforms.Resize(RESIZED), # redimensiona as imagens
        transforms.ToTensor(), # converte as imagens para tensores
        normalize
    ])

    # Cria o conjunto de dados de teste
    test_dataset = ImageFolder(DATASET_PATH / "test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_model(model, test_loader)