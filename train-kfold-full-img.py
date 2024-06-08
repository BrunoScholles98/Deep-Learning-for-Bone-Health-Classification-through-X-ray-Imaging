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

import utils

os.system('cls' if os.name == 'nt' else 'clear')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Caracteristicas do Treinamento
MODEL = 'efficientnet-b1'
BATCH_SIZE = 4
EPOCHS = 100
LOG_INTERVAL = 10
PERS_RESIZE_NUM = 3
REDUCELRONPLATEAU = True
PERSONALIZED_RESIZE = True
BETAS_LR = (0.9, 0.999)  # Valores padrão, mas você pode ajustá-los se desejar
NFOLDS = 5

# Paths
DATASET_PATH = Path('/d01/scholles/gigasistemica/datasets/KFOLD_DATASETS/UNITED_RB_CVAT_FULL_IMG_C1_C3')
TRAIN_NAME = utils.generate_training_name(MODEL, DATASET_PATH, BATCH_SIZE, EPOCHS)
OUTPUT_PATH = Path('/d01/scholles/gigasistemica/saved_models/K-Fold-Paper/' + TRAIN_NAME)
MODEL_SAVING_PATH = OUTPUT_PATH.joinpath(TRAIN_NAME + '.pth')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOG = OUTPUT_PATH / 'log'
STATS_PATH = OUTPUT_PATH / 'stats.txt'
IMG_RESIZE_PATH = '/d01/scholles/gigasistemica/datasets/CVAT_train/augmented/AUG_NEW_RB_CVAT_Train_FULL_IMG_C1_C3/test/C3/OPHUB2017-0046.jpg'

NUM_CLASSES = len([subfolder for subfolder in (DATASET_PATH).iterdir() if subfolder.is_dir()])

RESIZE = utils.train_resize(MODEL, PERSONALIZED_RESIZE)
print("Resize:",RESIZE)

class CustomDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img, label = self.data_list[index]

        # Aplica transformações, se fornecidas
        if self.transform:
            img = self.transform(img)

        # Converte a label para um tensor, se necessário
        label = torch.tensor(label)

        return img, label

def validate_model(model, criterion, val_dl, all_steps_counter_val, writer):
    accuracy_fnc = Accuracy().to(DEVICE)
    mean_loss_validation = 0
    val_epoch_accuracy = 0

    validation_bar = tqdm.tqdm(enumerate(val_dl), total=len(val_dl))
    validation_bar.set_description('Fold ' + str(actual_fold) + " - Validation Progress (Epoch)")    

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
    accuracy_fnc = Accuracy().to(DEVICE)
    mean_loss_train = 0
    train_epoch_accuracy = 0

    training_bar = tqdm.tqdm(enumerate(train_dl), total=len(train_dl))
    training_bar.set_description('Fold ' + str(actual_fold) + " - Training Progress (Epoch)")

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


def run_train_on_all_epochs(model, criterion, optimizer, scheduler, train_dl, val_dl, writer, actual_fold_path):
    epoch_bar = tqdm.tqdm(range(EPOCHS), initial=0, total=EPOCHS)
    epoch_bar.set_description('Fold ' + str(actual_fold) + " - Overall Progress")
    
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
            torch.save(model.state_dict(), actual_fold_path.joinpath(TRAIN_NAME + '.pth'))
        
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
            
    return y_true, y_pred

def calculate_auc_and_specificity(kfold_y_true, kfold_y_pred):
    # Calculate AUC Score
    auc_score = roc_auc_score(kfold_y_true, kfold_y_pred)

    # Calculate Specificity
    # Specificity = TN / (TN + FP)
    true_negatives = sum((1 for true, pred in zip(kfold_y_true, kfold_y_pred) if true == 0 and pred == 0))
    false_positives = sum((1 for true, pred in zip(kfold_y_true, kfold_y_pred) if true == 0 and pred == 1))
    specificity = true_negatives / (true_negatives + false_positives)

    return auc_score, specificity   

def main():
    kfold_y_true = []
    kfold_y_pred = []
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
   
    transform = transforms.Compose([ # Define as transformações que serão aplicadas às imagens
        transforms.Resize(RESIZE), # redimensiona as imagens
        transforms.ToTensor(), # converte as imagens para tensores
        normalize
        #transforms.Normalize(equilized_values())
    ])    

    print('Iniciada a leitura dos dados...')
    # Cria o conjunto de dados
    full_dataset = ImageFolder(DATASET_PATH, transform=None)

    kfold = KFold(n_splits=NFOLDS, shuffle=True)
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(full_dataset)):        
        
        global actual_fold
        actual_fold = fold + 1
        writer = SummaryWriter(TENSORBOARD_LOG.with_name(f"log_kfold_{actual_fold}"))    
        actual_fold_path =  OUTPUT_PATH / f"log_kfold_{actual_fold}"
         
        model = utils.load_model(MODEL, NUM_CLASSES)
        model.to(DEVICE)

        # Definir a função de perda e o otimizador
        criterion = torch.nn.CrossEntropyLoss()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=BETAS_LR)
        if REDUCELRONPLATEAU == True:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=1000000)

        # Sample elements randomly from a given list of ids, no replacement.
        train_data = [full_dataset[i] for i in train_ids]
        test_data = [full_dataset[i] for i in test_ids]
        train_data = utils.apply_augmentation(train_data)
        
        random.shuffle(train_data)
        random.shuffle(test_data)
        
        selected_train_subset = CustomDataset(train_data, transform=transform)
        selected_test_val_subset= CustomDataset(test_data, transform=transform)
        
        train_loader = DataLoader(selected_train_subset, batch_size=BATCH_SIZE)
        val_test_loader = DataLoader(selected_test_val_subset, batch_size=BATCH_SIZE)
        
        
        # Roda o treinamento e validação
        run_train_on_all_epochs(model, criterion, optimizer, scheduler, train_loader, val_test_loader, writer, actual_fold_path)
        
        y_true, y_pred = test_model(model, val_test_loader)
        kfold_y_true = kfold_y_true + y_true
        kfold_y_pred = kfold_y_pred + y_pred
    
    print('\n\n')
    report = classification_report(kfold_y_true, kfold_y_pred, digits=4)
    print(report)
    print('\n\n')

    auc_score, specificity = calculate_auc_and_specificity(kfold_y_true, kfold_y_pred)    
    report += f"\n\nAUC Score: {auc_score}\nSpecificity: {specificity}"    
    
    # Abra um arquivo de texto para escrita    
    with open(STATS_PATH, 'w') as arquivo:
        # Escreva a saída do relatório no arquivo
        arquivo.write(report)    

    torch.save(model.state_dict(), actual_fold_path.joinpath(TRAIN_NAME + '.pth'))
    print("Arquivo salvo com sucesso!")
    
    GT_vs_Predictions = {
        "kfold_y_true": kfold_y_true,
        "kfold_y_pred": kfold_y_pred
    }
    
    GT_vs_Predictions_json_path = OUTPUT_PATH / 'GT_vs_Predictions.json'    
    # Salvar os dados em um arquivo JSON
    with open(GT_vs_Predictions_json_path, 'w') as f:
        json.dump(GT_vs_Predictions, f)
        
if __name__ == '__main__':
    main()