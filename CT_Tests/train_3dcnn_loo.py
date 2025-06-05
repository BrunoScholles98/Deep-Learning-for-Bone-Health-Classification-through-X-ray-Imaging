
"""
Leave-One-Out training script — SEM early stopping, SEM curva de treino,
SEM matriz de confusão por fold. Ao final grava:
  • confusion_matrix_overall.png
  • loo_metrics.txt   (Accuracy, Precision, Recall, F1, AUC, Specificity, NPV)
"""

import os
import random
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torchvision import transforms
from torchvision.models.video import r3d_18
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import logging

DATA_DIR      = '/mnt/ssd/brunoscholles/GigaSistemica/Datasets/TM_3D_64Stacks_Sagital'
OUTPUT_DIR    = '/mnt/ssd/brunoscholles/GigaSistemica/Models/TC_Models'
MODEL_NAME    = 'r3d_18'

NUM_CLASSES   = 2
EPOCHS        = 50
BATCH_SIZE    = 2
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
SEED          = 42
IMG_SIZE      = 112
MAX_SLICES    = 64
AUG_INTENSITY = 0.5
USE_AMP       = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class Resize3D:
    def __init__(self, size):
        self.size = size  # (H,W)
    def __call__(self, vol):
        vol = vol.unsqueeze(0)
        vol = F.interpolate(vol, size=(vol.shape[2], *self.size),
                            mode="trilinear", align_corners=False)
        return vol.squeeze(0)

class RandomFlip3D:
    def __init__(self, prob=0.5, dims=(2, 3)):
        self.prob = prob
        self.dims = dims
    def __call__(self, vol):
        if random.random() < self.prob:
            for d in self.dims:
                vol = torch.flip(vol, dims=[d])
        return vol

class Normalize3D:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, vol):
        mean_t = torch.tensor(self.mean, dtype=vol.dtype,
                              device=vol.device).view(-1,1,1,1)
        std_t  = torch.tensor(self.std,  dtype=vol.dtype,
                              device=vol.device).view(-1,1,1,1)
        return (vol - mean_t) / std_t

class MedicalVolumeAugmentations3D:
    def __init__(self, img_size, intensity=0.5):
        self.train_tf = transforms.Compose([
            RandomFlip3D(prob=intensity),
            Resize3D((img_size, img_size)),
            Normalize3D([0.43216, 0.394666, 0.37645],
                        [0.22803, 0.22145, 0.216989]),
        ])
        self.val_tf = transforms.Compose([
            Resize3D((img_size, img_size)),
            Normalize3D([0.43216, 0.394666, 0.37645],
                        [0.22803, 0.22145, 0.216989]),
        ])
    def get(self, train=True):
        return self.train_tf if train else self.val_tf

class MedicalImageStackDataset(Dataset):
    def __init__(self, data_dir, split, transform, max_slices=64):
        self.transform = transform
        self.max_slices = max_slices
        self.samples = []

        split_dir = os.path.join(data_dir, split)
        for class_name in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_name)
            if split == 'test' and class_name == '1' and \
               os.path.isdir(os.path.join(class_path, 'i')):
                class_path = os.path.join(class_path, 'i')
            if not os.path.isdir(class_path):
                continue
            label = int(class_name)
            self.samples += [(os.path.join(class_path, f), label)
                             for f in os.listdir(class_path) if f.endswith('.npy')]
        logger.info(f'Loaded {split}: {len(self.samples)} samples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        vol = np.load(path).transpose(2,0,1)                # (D,H,W)
        d = vol.shape[0]
        if d > self.max_slices:
            s = (d-self.max_slices)//2
            vol = vol[s:s+self.max_slices]
        elif d < self.max_slices:
            vol = np.pad(vol, ((0,self.max_slices-d),(0,0),(0,0)))

        vol = (torch.from_numpy(vol)
                 .float().unsqueeze(0)       # (1,D,H,W)
                 .repeat(3,1,1,1))           # (3,D,H,W)
        vol = self.transform(vol) if self.transform else vol
        return vol, label

class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        x,y = self.subset[idx]
        return (self.transform(x) if self.transform else x), y

def create_model(num_classes):
    m = r3d_18(pretrained=True)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def train_one_split(model, train_loader, val_loader, criterion, optimizer,
                    scheduler, device, epochs, use_amp=False):
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    best_state, best_f1 = None, 0.0

    for ep in range(epochs):
        logger.info(f'  Epoch {ep+1}/{epochs}')
        model.train()
        for x,y in tqdm(train_loader, leave=False):
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    out = model(x); loss = criterion(out,y)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
            else:
                out = model(x); loss = criterion(out,y)
                loss.backward(); optimizer.step()

        # validação
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x,y in val_loader:
                prob = torch.softmax(model(x.to(device)),1).cpu()
                y_pred.append(prob.argmax().item()); y_true.append(y.item())
        f1 = f1_score(y_true, y_pred, zero_division=0)
        scheduler.step(f1)

        if f1 > best_f1:
            best_f1, best_state = f1, model.state_dict()

    model.load_state_dict(best_state)
    return model

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
    plt.savefig(save_path); plt.close()

def run_leave_one_out():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(OUTPUT_DIR, f'LOO_{MODEL_NAME}_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    aug = MedicalVolumeAugmentations3D(IMG_SIZE, AUG_INTENSITY)
    train_raw = MedicalImageStackDataset(DATA_DIR,'train',None,MAX_SLICES)
    test_raw  = MedicalImageStackDataset(DATA_DIR,'test', None,MAX_SLICES)
    full_ds   = ConcatDataset([train_raw, test_raw])
    logger.info(f'LOO em {len(full_ds)} amostras.')

    counts = Counter(lbl for _,lbl in sum([d.samples for d in [train_raw,test_raw]], []))
    tot = sum(counts.values())
    cls_w = torch.tensor([tot/(len(counts)*counts[i]) for i in range(NUM_CLASSES)])

    global_true, global_pred, global_prob = [], [], []

    for idx in range(len(full_ds)):
        logger.info(f'\n========== LOO {idx+1}/{len(full_ds)} ==========')

        val_raw   = Subset(full_ds, [idx])
        train_raw = Subset(full_ds, [i for i in range(len(full_ds)) if i!=idx])

        train_ds = SubsetWithTransform(train_raw, aug.get(True))
        val_ds   = SubsetWithTransform(val_raw,   aug.get(False))

        train_ld = DataLoader(train_ds, BATCH_SIZE, True, pin_memory=torch.cuda.is_available())
        val_ld   = DataLoader(val_ds,   1, False, pin_memory=torch.cuda.is_available())

        model = create_model(NUM_CLASSES).to(device)
        crit  = nn.CrossEntropyLoss(weight=cls_w.to(device))
        opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sch   = optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', factor=0.1, patience=5)

        model = train_one_split(model, train_ld, val_ld, crit, opt, sch, device,
                                EPOCHS, USE_AMP)

        # predição do único exemplo de validação
        model.eval()
        x_val, y_val = val_ds[0]
        with torch.no_grad():
            prob = torch.softmax(model(x_val.unsqueeze(0).to(device)),1).cpu().squeeze(0)
        pred = prob.argmax().item()

        global_true.append(y_val)
        global_pred.append(pred)
        global_prob.append(prob.numpy())

    cm = confusion_matrix(global_true, global_pred, labels=[0,1])
    plot_confusion_matrix(cm, [str(i) for i in range(NUM_CLASSES)],
                          os.path.join(out_dir,'confusion_matrix_overall.png'))

    tn,fp,fn,tp = cm.ravel()
    accuracy  = accuracy_score(global_true, global_pred)
    precision = precision_score(global_true, global_pred, zero_division=0)
    recall    = recall_score(global_true, global_pred, zero_division=0)
    f1        = f1_score(global_true, global_pred, zero_division=0)
    specificity = tn/(tn+fp) if (tn+fp) else 0.0
    npv         = tn/(tn+fn) if (tn+fn) else 0.0
    auc         = roc_auc_score(global_true, np.vstack(global_prob)[:,1])

    with open(os.path.join(out_dir,'loo_metrics.txt'),'w') as f:
        f.write(f'Accuracy   : {accuracy:.4f}\n')
        f.write(f'Precision  : {precision:.4f}\n')
        f.write(f'Recall     : {recall:.4f}\n')
        f.write(f'Specificity: {specificity:.4f}\n')
        f.write(f'NPV        : {npv:.4f}\n')
        f.write(f'F1-Score   : {f1:.4f}\n')
        f.write(f'AUC        : {auc:.4f}\n')

    logger.info('\n=== Desempenho LOO ===')
    logger.info(open(os.path.join(out_dir,'loo_metrics.txt')).read().strip())

if __name__ == '__main__':
    run_leave_one_out()