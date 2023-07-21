import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from torch.autograd import Variable

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

from PIL import Image
import tqdm

from old.older_trainings.ResNet import ResNet50, ResNet152

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Caracteristicas do Treinamento
NUM_CLASSES = 2
TRAIN_NAME = 'Sexto_ResNet50_Batch8'
BATCH_SIZE = 8
EPOCHS = 300
LOG_INTERVAL = 10

# Paths
DATASET_PATH = Path('/d01/scholles/other/gigasistemica_sandbox_scholles/dataset')
OUTPUT_PATH = Path('/d01/scholles/other/gigasistemica_sandbox_scholles/results/' + TRAIN_NAME)

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOG = OUTPUT_PATH / 'log'
STATS_PATH = OUTPUT_PATH / 'stats.txt'

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
            y_hat_copy = y_hat.clone().detach()
            y_hat = torch.argmax(y_hat, dim=1).float()
            labels_copy = labels.clone().detach()
            labels = labels.float()
            loss_val = criterion(y_hat, labels)
            mean_loss_validation += loss_val
            val_iteration_accuracy = accuracy_fnc(y_hat_copy, labels_copy)
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
    y_hat = torch.argmax(y_hat, dim=1).float()
    labels = labels.float()
    loss = criterion(y_hat, labels)
    loss = Variable(loss, requires_grad = True)
    loss.backward()
    optimizer.step()

    return y_hat, loss


def train_by_one_epoch(model, criterion, optimizer, train_dl, all_steps_counter_train, writer):
    accuracy_fnc = Accuracy().to(DEVICE)
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
    epoch_bar.set_description("Training Progress (Overall)")
    
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

        scheduler.step(mean_loss_train)


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

    target_names = ["Healthy", "Osteop."]
    
    report = classification_report(y_true, y_pred)
    print(report)

    # Abra um arquivo de texto para escrita
    with open(STATS_PATH, 'w') as arquivo:
        # Escreva a saída do relatório no arquivo
        arquivo.write(report)

    # Confirme que o arquivo foi salvo
    print("Arquivo salvo com sucesso!")


def get_resized_size():
    path = '/d01/scholles/other/gigasistemica_sandbox_scholles/dataset/test/Healthy'
    files = os.listdir(path)
    image = Image.open(os.path.join(path, files[0]))
    w, h = image.size
    resized = (int(w/4), int(h/4))
    return resized


def main():    
    resized = get_resized_size()    
    transform = transforms.Compose([ # Define as transformações que serão aplicadas às imagens
        transforms.Resize(resized), # redimensiona as imagens
        transforms.ToTensor() # converte as imagens para tensores
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

    model = ResNet50(NUM_CLASSES).to(DEVICE)

    criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)

    run_train_on_all_epochs(model, criterion, optimizer, scheduler, train_loader, val_loader)    

    print('\n\nClassification Report')
    test_model(model, test_loader)
    print('\n\n')

if __name__ == '__main__':
    main()    