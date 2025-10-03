import os
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit  # <- para splits estratificados
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

# Características do Treinamento
MODEL = args.model
BATCH_SIZE = 4
EPOCHS = 100
LOG_INTERVAL = 10
PERS_RESIZE_NUM = 3
REDUCELRONPLATEAU = True
PERSONALIZED_RESIZE = True
SEED = 42  # para reprodutibilidade

# Seeds (opcional mas recomendado)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
try:
    import numpy as np
    np.random.seed(SEED)
except Exception:
    pass

# Paths
# Agora DATASET_PATH aponta para a PASTA RAIZ com subpastas = classes (sem train/val/test)
DATASET_PATH = Path('/mnt/nas/BrunoScholles/Gigasistemica/Datasets/OP_Rolling_Ball_Imgs')
TRAIN_NAME = utils.generate_training_name(MODEL, DATASET_PATH, BATCH_SIZE, EPOCHS)
OUTPUT_PATH = Path('/mnt/nas/BrunoScholles/Gigasistemica/Models/test_efficient/' + TRAIN_NAME)
MODEL_SAVING_PATH = OUTPUT_PATH.joinpath(TRAIN_NAME + '.pth')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOG = OUTPUT_PATH / 'log'
STATS_PATH = OUTPUT_PATH / 'stats.txt'

RESIZE = utils.train_resize(MODEL, PERSONALIZED_RESIZE)
print("Resize:", RESIZE)

writer = SummaryWriter(TENSORBOARD_LOG)
NUM_CLASSES = None

def validate_model(model, criterion, val_dl, all_steps_counter_val, writer):
    accuracy_fnc = Accuracy(task='multiclass', num_classes=NUM_CLASSES).to(DEVICE)
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
    accuracy_fnc = Accuracy(task='multiclass', num_classes=NUM_CLASSES).to(DEVICE)
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

        # Checkpoint a cada 10 épocas
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

    with open(STATS_PATH, 'w') as arquivo:
        arquivo.write(report)

    print("Arquivo salvo com sucesso!")


def make_stratified_splits(dataset, seed=42):
    """
    Retorna índices (train_idx, val_idx, test_idx) com split estratificado 60/20/20.
    """
    targets = dataset.targets  # lista de rótulos por amostra
    indices = list(range(len(targets)))

    # 1) Train (60%) vs Temp (40%)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
    train_idx, temp_idx = next(sss1.split(indices, targets))

    # 2) Val (20% total) vs Test (20% total) a partir do Temp (metade/metade)
    temp_targets = [targets[i] for i in temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_rel, test_rel = next(sss2.split(list(range(len(temp_idx))), temp_targets))
    val_idx = [temp_idx[i] for i in val_rel]
    test_idx = [temp_idx[i] for i in test_rel]

    return train_idx, val_idx, test_idx


def main():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.ToTensor(),
        normalize
    ])

    print('Iniciada a leitura dos dados a partir da pasta raiz (classes como subpastas)...')

    # Dataset único (apenas raiz com subpastas de classes)
    full_dataset = ImageFolder(DATASET_PATH, transform=transform)
    global NUM_CLASSES
    NUM_CLASSES = len(full_dataset.classes)
    print(f'Classes ({NUM_CLASSES}): {full_dataset.classes}')

    # Splits estratificados 60/20/20
    train_idx, val_idx, test_idx = make_stratified_splits(full_dataset, seed=SEED)

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = utils.load_model(MODEL, NUM_CLASSES)
    model.to(DEVICE)

    # Loss / Otimizador / Scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if REDUCELRONPLATEAU:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10000)

    # Treino + Val
    run_train_on_all_epochs(model, criterion, optimizer, scheduler, train_loader, val_loader)

    print('\n\nClassification Report')
    test_model(model, test_loader)
    print('\n\n')

    # Salvar o modelo treinado
    torch.save(model.state_dict(), MODEL_SAVING_PATH)


if __name__ == '__main__':
    main()