import os
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import random
import tqdm
import json
import argparse

import utils

os.system('cls' if os.name == 'nt' else 'clear')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

parser = argparse.ArgumentParser(description='Configurações do treinamento.')
parser.add_argument('--model', type=str, help='Nome do modelo')
args = parser.parse_args()

# Caracteristicas do Treinamento
MODEL = args.model
BATCH_SIZE = 8
EPOCHS = 100
LOG_INTERVAL = 5
PERS_RESIZE_NUM = 3
REDUCELRONPLATEAU = True
PERSONALIZED_RESIZE = False
BETAS_LR = (0.9, 0.999)  # Valores padrão, mas você pode ajustá-los se desejar

# Paths
DATASET_PATH = Path('/d01/scholles/gigasistemica/datasets/SIMPLE_TRAIN/CROPPED_PR')
TRAIN_NAME = utils.generate_training_name(MODEL, DATASET_PATH, BATCH_SIZE, EPOCHS)
OUTPUT_PATH = Path('/d01/scholles/gigasistemica/saved_models/Full_Dataset_Train_Paper/CROPPED_PR/' + TRAIN_NAME)
MODEL_SAVING_PATH = OUTPUT_PATH.joinpath(TRAIN_NAME + '_test.pth')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOG = OUTPUT_PATH / 'log'
STATS_PATH = OUTPUT_PATH / 'stats.txt'

# Caracteristicas do Treinamento
NUM_CLASSES = len([subfolder for subfolder in (DATASET_PATH / 'train').iterdir() if subfolder.is_dir()])

RESIZE = utils.train_resize(MODEL, PERSONALIZED_RESIZE)
print("Resize:",RESIZE)

writer = SummaryWriter(TENSORBOARD_LOG)

def validate_model(model, criterion, val_dl, all_steps_counter_val, writer):
    accuracy_fnc = Accuracy().to(DEVICE)
    mean_loss_validation = 0
    val_epoch_accuracy = 0

    validation_bar = tqdm.tqdm(enumerate(val_dl), total=len(val_dl))
    validation_bar.set_description("Validation Progress (Epoch)")    

    with torch.no_grad():
        for validation_step, inp in validation_bar:
            inputs, labels = inp
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            y_hat = model(inputs)
            loss_val = criterion(y_hat, labels)
            mean_loss_validation += loss_val
            val_iteration_accuracy = accuracy_fnc(y_hat, labels)
            val_epoch_accuracy += val_iteration_accuracy

            if validation_step % LOG_INTERVAL == 0:
                writer.add_scalar('Validation/Iteration_Loss', loss_val, all_steps_counter_val)
                writer.add_scalar('Validation/Iteration_Accuracy', val_iteration_accuracy, all_steps_counter_val)
                
            all_steps_counter_val += 1
        
        mean_loss_validation /= len(val_dl)
        val_epoch_accuracy /= len(val_dl)

    return all_steps_counter_val, mean_loss_validation, val_epoch_accuracy


def train_one_step(model, optimizer, criterion, inputs, labels):
    optimizer.zero_grad()                
    y_hat = model(inputs)
    loss = criterion(y_hat, labels)
    loss.backward()
    optimizer.step()

    return y_hat, loss


def train_by_one_epoch(model, criterion, optimizer, train_dl, all_steps_counter_train, writer):
    accuracy_fnc = Accuracy(task='BINARY').to(DEVICE)
    mean_loss_train = 0
    train_epoch_accuracy = 0

    training_bar = tqdm.tqdm(enumerate(train_dl), total=len(train_dl))
    training_bar.set_description("Training Progress (Epoch)")

    for step_train, inp in training_bar:
        inputs, labels = inp
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        y_hat, loss = train_one_step(model, optimizer, criterion, inputs, labels)
        mean_loss_train += loss
        training_iteration_accuracy = accuracy_fnc(y_hat, labels)
        train_epoch_accuracy += training_iteration_accuracy
            
        if step_train % LOG_INTERVAL == 0:
            writer.add_scalar('Train/Iteration_Loss', loss, all_steps_counter_train)
            writer.add_scalar('Train/Iteration_Accuracy', training_iteration_accuracy, all_steps_counter_train)
        
        all_steps_counter_train += 1

    mean_loss_train /= len(train_dl)
    train_epoch_accuracy /= len(train_dl)
    
    return all_steps_counter_train, mean_loss_train, train_epoch_accuracy


def run_train_on_all_epochs(model, criterion, optimizer, scheduler, train_dl, val_dl):
    writer = SummaryWriter(TENSORBOARD_LOG)
    epoch_bar = tqdm.tqdm(range(EPOCHS), initial=0, total=EPOCHS)
    epoch_bar.set_description("Overall Progress")
    
    all_steps_counter_train = 0
    all_steps_counter_val = 0

    for epoch in epoch_bar:

        all_steps_counter_train, mean_loss_train, train_epoch_accuracy = \
            train_by_one_epoch(model, criterion, optimizer, train_dl, all_steps_counter_train, writer)
        all_steps_counter_val, mean_loss_validation, val_epoch_accuracy = \
            validate_model(model, criterion, val_dl, all_steps_counter_val, writer)

        writer.add_scalar('Train/Epoch_Loss', mean_loss_train, epoch)
        writer.add_scalar('Train/Epoch_Accuracy', train_epoch_accuracy, epoch)        
        writer.add_scalar('Validation/Epoch_Loss', mean_loss_validation, epoch)
        writer.add_scalar('Validation/Epoch_Accuracy', val_epoch_accuracy, epoch)

        # Checkpoint a cada 10 epocas
        if epoch % 10 == 0:
            torch.save(model.state_dict(), MODEL_SAVING_PATH)
        
        scheduler.step(mean_loss_train)
        
        os.system('cls' if os.name == 'nt' else 'clear')


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

    # Abra um arquivo de texto para escrita
    with open(STATS_PATH, 'w') as arquivo:
        # Escreva a saída do relatório no arquivo
        arquivo.write(report)

    # Confirme que o arquivo foi salvo
    print("Arquivo salvo com sucesso!")
    

def main():   
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
   
    transform = transforms.Compose([ # Define as transformações que serão aplicadas às imagens
        transforms.Resize(RESIZE), # redimensiona as imagens
        transforms.ToTensor(), # converte as imagens para tensores
        normalize
        #transforms.Normalize(equilized_values())
    ])

    print('Iniciada a leitura dos dados...')
    # Cria o conjunto de dados de treinamento
    train_dataset = ImageFolder(DATASET_PATH / "train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Cria o conjunto de dados de validação
    val_dataset = ImageFolder(DATASET_PATH / "val", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Cria o conjunto de dados de teste
    test_dataset = ImageFolder(DATASET_PATH / "test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = utils.load_model(MODEL, NUM_CLASSES)
    model.to(DEVICE)

    # Definir a função de perda e o otimizador
    criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=BETAS_LR)
    if REDUCELRONPLATEAU == True:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=10000)    

    # Roda o treinamento e validação
    run_train_on_all_epochs(model, criterion, optimizer, scheduler, train_loader, val_loader) 
    
    print('\n\nClassification Report')
    test_model(model, test_loader)
    print('\n\n')

    # Salvar o modelo treinado    
    torch.save(model.state_dict(), MODEL_SAVING_PATH)

if __name__ == '__main__':
    main()    