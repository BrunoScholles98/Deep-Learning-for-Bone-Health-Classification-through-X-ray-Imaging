import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
import random
import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import utils

# ==========================================
# CONFIGURATION
# ==========================================
# Set these variables to configure the training
USE_CROPPED_IMAGES = False   # Set to True for cropped images, False for full images
USE_KFOLD = False            # Set to True for K-Fold CV, False for simple train/test split

# Hyperparameters
BATCH_SIZE = 8 if USE_CROPPED_IMAGES else 4
EPOCHS = 100
LR = 0.001
BETAS_LR = (0.9, 0.999)
REDUCELRONPLATEAU = True
SEED = 42
NFOLDS = 5
LOG_INTERVAL = 10

# Device Configuration
GPU_IDS = [0] # Select GPU IDs, e.g., [0] or [0, 1]

if torch.cuda.is_available() and GPU_IDS:
    DEVICE = torch.device(f'cuda:{GPU_IDS[0]}')
else:
    DEVICE = torch.device('cpu')

print(f"Using device: {DEVICE}")
if len(GPU_IDS) > 1:
    print(f"Using {len(GPU_IDS)} GPUs: {GPU_IDS}")

# Model Configuration
# You can change the default model here or pass it via command line args if implemented
DEFAULT_MODEL = 'swin_base' if USE_CROPPED_IMAGES else 'efficientnet-b1'

# ==========================================
# PATHS AND CONSTANTS
# ==========================================
def get_dataset_path(use_cropped, use_kfold):
    if use_cropped:
        if use_kfold:
            return Path('/d01/scholles/gigasistemica/datasets/KFOLD_DATASETS_CVAT/RB_CVAT_Cropped_600x600_C1_C3')
        else:
            return Path('/d01/scholles/gigasistemica/datasets/SIMPLE_TRAIN/CROPPED_PR')
    else:
        if use_kfold:
            return Path('/d01/scholles/gigasistemica/datasets/KFOLD_DATASETS/UNITED_RB_CVAT_FULL_IMG_C1_C3')
        else:
            return Path('/mnt/nas/BrunoScholles/Gigasistemica/Datasets/OP_Rolling_Ball_Imgs')

DATASET_PATH = get_dataset_path(USE_CROPPED_IMAGES, USE_KFOLD)

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help='Model name')
args = parser.parse_args()
MODEL = args.model

# Generate Training Name and Output Path
TRAIN_NAME = utils.generate_training_name(MODEL, DATASET_PATH, BATCH_SIZE, EPOCHS)
if USE_KFOLD:
    OUTPUT_PATH = Path('/mnt/nas/BrunoScholles/Gigasistemica/Models/kfold_efficient/' + TRAIN_NAME)
else:
    OUTPUT_PATH = Path('/mnt/nas/BrunoScholles/Gigasistemica/Models/test_efficient/' + TRAIN_NAME)

if not OUTPUT_PATH.exists():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

TENSORBOARD_LOG = OUTPUT_PATH / 'tensorboard_log'
STATS_PATH = OUTPUT_PATH / 'stats.txt'
MODEL_SAVING_PATH = OUTPUT_PATH.joinpath(TRAIN_NAME + '.pth')

# Resize Logic
PERSONALIZED_RESIZE = not USE_CROPPED_IMAGES # Full images use personalized resize (449, 954)
RESIZE = utils.train_resize(MODEL, PERSONALIZED_RESIZE)

print(f"Configuration: Cropped={USE_CROPPED_IMAGES}, KFold={USE_KFOLD}")
print(f"Dataset Path: {DATASET_PATH}")
print(f"Output Path: {OUTPUT_PATH}")
print(f"Resize Dimensions: {RESIZE}")
print(f"Model: {MODEL}")

# ==========================================
# HELPER CLASSES AND FUNCTIONS
# ==========================================

class CustomDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

def make_stratified_splits(dataset, seed=42):
    """
    Returns indices (train_idx, val_idx, test_idx) with 60/20/20 stratified split.
    Used for Simple + Full Image case.
    """
    targets = dataset.targets
    indices = list(range(len(targets)))

    # Train (60%) vs Temp (40%)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
    train_idx, temp_idx = next(sss1.split(indices, targets))

    # Val (20%) vs Test (20%) from Temp split
    temp_targets = [targets[i] for i in temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_rel, test_rel = next(sss2.split(list(range(len(temp_idx))), temp_targets))
    val_idx = [temp_idx[i] for i in val_rel]
    test_idx = [temp_idx[i] for i in test_rel]

    return train_idx, val_idx, test_idx

def calculate_auc_and_specificity(y_true, y_pred):
    try:
        auc_score = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc_score = 0.0  # Only one class present

    # Specificity = TN / (TN + FP)
    true_negatives = sum((1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 0))
    false_positives = sum((1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1))
    
    if (true_negatives + false_positives) > 0:
        specificity = true_negatives / (true_negatives + false_positives)
    else:
        specificity = 0.0

    return auc_score, specificity

def accuracy_fnc(y_true, y_pred):
    _, predicted = torch.max(y_true.data, 1)
    total = y_pred.size(0)
    correct = (predicted == y_pred).sum().item()
    return correct / total

def run_train_on_all_epochs(model, criterion, optimizer, scheduler, train_loader, val_loader, writer=None, fold_path=None):
    if writer is None:
        writer = SummaryWriter(TENSORBOARD_LOG)
        
    all_steps_counter_train = 0
    all_steps_counter_val = 0
    
    for epoch in range(EPOCHS):
        model.train()
        mean_loss_train = 0
        train_epoch_accuracy = 0
        
        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        if fold_path:
            train_bar.set_description(f'Fold {actual_fold} - Training Progress (Epoch {epoch+1}/{EPOCHS})')
        else:
            train_bar.set_description(f'Training Progress (Epoch {epoch+1}/{EPOCHS})')

        for step, (inputs, labels) in train_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            y_hat = model(inputs)
            loss = criterion(y_hat, labels)
            loss.backward()
            optimizer.step()
            
            mean_loss_train += loss.item()
            train_iteration_accuracy = accuracy_fnc(y_hat, labels)
            train_epoch_accuracy += train_iteration_accuracy
            
            if step % LOG_INTERVAL == 0:
                writer.add_scalar('Train/Iteration_Loss', loss.item(), all_steps_counter_train)
                writer.add_scalar('Train/Iteration_Accuracy', train_iteration_accuracy, all_steps_counter_train)
            
            all_steps_counter_train += 1
            
        mean_loss_train /= len(train_loader)
        train_epoch_accuracy /= len(train_loader)
        
        # Validation
        model.eval()
        mean_loss_validation = 0
        val_epoch_accuracy = 0
        
        val_bar = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)
        if fold_path:
             val_bar.set_description(f'Fold {actual_fold} - Validation Progress (Epoch {epoch+1}/{EPOCHS})')
        else:
             val_bar.set_description(f'Validation Progress (Epoch {epoch+1}/{EPOCHS})')

        with torch.no_grad():
            for step, (inputs, labels) in val_bar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                y_hat = model(inputs)
                loss_val = criterion(y_hat, labels)
                mean_loss_validation += loss_val.item()
                val_iteration_accuracy = accuracy_fnc(y_hat, labels)
                val_epoch_accuracy += val_iteration_accuracy

                if step % LOG_INTERVAL == 0:
                    writer.add_scalar('Validation/Iteration_Loss', loss_val.item(), all_steps_counter_val)
                    writer.add_scalar('Validation/Iteration_Accuracy', val_iteration_accuracy, all_steps_counter_val)
                    
                all_steps_counter_val += 1
        
        mean_loss_validation /= len(val_loader)
        val_epoch_accuracy /= len(val_loader)

        writer.add_scalar('Train/Epoch_Loss', mean_loss_train, epoch)
        writer.add_scalar('Train/Epoch_Accuracy', train_epoch_accuracy, epoch)
        writer.add_scalar('Validation/Epoch_Loss', mean_loss_validation, epoch)
        writer.add_scalar('Validation/Epoch_Accuracy', val_epoch_accuracy, epoch)
        
        if REDUCELRONPLATEAU:
            scheduler.step(mean_loss_validation)
        
        # Save checkpoints
        if fold_path:
             pass  # KFold saves at end of fold
        else:
             if epoch % 10 == 0:
                torch.save(model.state_dict(), MODEL_SAVING_PATH)

def test_model(model, test_loader):
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            y_true += labels.cpu().numpy().tolist()
            y_pred += predicted.cpu().numpy().tolist()
            
    return y_true, y_pred

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    # Transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
   
    transform = transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.ToTensor(),
        normalize
    ])    

    print('Loading dataset...')
    
    # K-Fold cross-validation
    if USE_KFOLD:
        full_dataset = ImageFolder(DATASET_PATH, transform=None)
        kfold = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
        
        kfold_y_true = []
        kfold_y_pred = []
        
        for fold, (train_ids, test_ids) in enumerate(kfold.split(full_dataset)):        
            global actual_fold
            actual_fold = fold + 1
            print(f"Starting Fold {actual_fold}/{NFOLDS}")
            
            writer = SummaryWriter(TENSORBOARD_LOG.with_name(f"log_kfold_{actual_fold}"))    
            actual_fold_path =  OUTPUT_PATH / f"log_kfold_{actual_fold}"
            actual_fold_path.mkdir(parents=True, exist_ok=True)
            
            model = utils.load_model(MODEL, len(full_dataset.classes))
            model.to(DEVICE)
            
            if len(GPU_IDS) > 1 and torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model, device_ids=GPU_IDS)

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS_LR)
            
            if REDUCELRONPLATEAU:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)
            else:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10000)

            train_data = [full_dataset[i] for i in train_ids]
            test_data = [full_dataset[i] for i in test_ids]
            
            # Apply augmentation to training data
            train_data = utils.apply_augmentation(train_data)
            
            random.shuffle(train_data)
            random.shuffle(test_data)
            
            selected_train_subset = CustomDataset(train_data, transform=transform)
            selected_test_val_subset= CustomDataset(test_data, transform=transform)
            
            train_loader = DataLoader(selected_train_subset, batch_size=BATCH_SIZE, shuffle=True)
            val_test_loader = DataLoader(selected_test_val_subset, batch_size=BATCH_SIZE, shuffle=False)
            
            run_train_on_all_epochs(model, criterion, optimizer, scheduler, train_loader, val_test_loader, writer, actual_fold_path)
            
            y_true, y_pred = test_model(model, val_test_loader)
            kfold_y_true += y_true
            kfold_y_pred += y_pred
            
            torch.save(model.state_dict(), actual_fold_path.joinpath(TRAIN_NAME + '.pth'))
            print(f"Fold {actual_fold} model saved.")

        print('\n\nK-Fold Classification Report')
        report = classification_report(kfold_y_true, kfold_y_pred, digits=4)
        print(report)
        
        auc_score, specificity = calculate_auc_and_specificity(kfold_y_true, kfold_y_pred)    
        report += f"\n\nAUC Score: {auc_score}\nSpecificity: {specificity}"    
        
        with open(STATS_PATH, 'w') as arquivo:
            arquivo.write(report)    
        
        GT_vs_Predictions = {
            "kfold_y_true": kfold_y_true,
            "kfold_y_pred": kfold_y_pred
        }
        
        with open(OUTPUT_PATH / 'GT_vs_Predictions.json', 'w') as f:
            json.dump(GT_vs_Predictions, f)

    # Simple train/val/test split
    else:
        if USE_CROPPED_IMAGES:
            train_dataset = ImageFolder(DATASET_PATH / "train", transform=transform)
            val_dataset = ImageFolder(DATASET_PATH / "val", transform=transform)
            test_dataset = ImageFolder(DATASET_PATH / "test", transform=transform)
            num_classes = len(train_dataset.classes)
        else:
            full_dataset = ImageFolder(DATASET_PATH, transform=transform)
            num_classes = len(full_dataset.classes)
            print(f'Classes ({num_classes}): {full_dataset.classes}')
            
            train_idx, val_idx, test_idx = make_stratified_splits(full_dataset, seed=SEED)
            train_dataset = Subset(full_dataset, train_idx)
            val_dataset = Subset(full_dataset, val_idx)
            test_dataset = Subset(full_dataset, test_idx)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = utils.load_model(MODEL, num_classes)
        model.to(DEVICE)

        if len(GPU_IDS) > 1 and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, device_ids=GPU_IDS)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        
        if REDUCELRONPLATEAU:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10000)

        run_train_on_all_epochs(model, criterion, optimizer, scheduler, train_loader, val_loader)

        print('\n\nClassification Report')
        y_true, y_pred = test_model(model, test_loader)
        report = classification_report(y_true, y_pred, digits=4)
        print(report)

        with open(STATS_PATH, 'w') as arquivo:
            arquivo.write(report)

        torch.save(model.state_dict(), MODEL_SAVING_PATH)
        print("Final model saved successfully!")

if __name__ == '__main__':
    main()
