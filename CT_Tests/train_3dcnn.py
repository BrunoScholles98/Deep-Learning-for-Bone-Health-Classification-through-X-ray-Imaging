import os
import sys
import argparse
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
)
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torchvision import transforms
from torchvision.models.video import r3d_18
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# ----------------------------------------------------------------------------
#                            GLOBAL VARIABLES
# ----------------------------------------------------------------------------
DATA_DIR = '/mnt/ssd/brunoscholles/GigaSistemica/Datasets/TM_3D_64Stacks_Coronal'  # Adjust to your path
MODEL_NAME = 'r3d_18'
NUM_CLASSES = 2
EPOCHS = 50
BATCH_SIZE = 2
LR = 1e-4
WEIGHT_DECAY = 1e-4
OUTPUT_DIR = '/mnt/ssd/brunoscholles/GigaSistemica/Models/TC_Models'
SEED = 42
IMG_SIZE = 112
STACK_METHOD = 'multi'  # Kept for compatibility
MAX_SLICES = 64         # Depth (number of slices) for the 3D CNN input
EARLY_STOPPING = 10
USE_AMP = False
AUG_INTENSITY = 0.5
RUN_LEAVE_ONE_OUT = True
# ----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

param_config = {
    "DATA_DIR": DATA_DIR,
    "MODEL_NAME": MODEL_NAME,
    "NUM_CLASSES": NUM_CLASSES,
    "EPOCHS": EPOCHS,
    "BATCH_SIZE": BATCH_SIZE,
    "LR": LR,
    "WEIGHT_DECAY": WEIGHT_DECAY,
    "OUTPUT_DIR": OUTPUT_DIR,
    "SEED": SEED,
    "IMG_SIZE": IMG_SIZE,
    "STACK_METHOD": STACK_METHOD,
    "MAX_SLICES": MAX_SLICES,
    "EARLY_STOPPING": EARLY_STOPPING,
    "USE_AMP": USE_AMP,
    "AUG_INTENSITY": AUG_INTENSITY
}

# ----------------------------------------------------------------------------
#                          Utility / Reproducibility
# ----------------------------------------------------------------------------

def set_seed(seed: int):
    """Freeze random generators for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ----------------------------------------------------------------------------
#       Function to unify the shape of the 3 views with zero‑padding
# ----------------------------------------------------------------------------

def unify_shape_2d(axial, coronal, sagittal):
    """Pad the three 2‑D views so they share the same H×W."""
    h_ax, w_ax = axial.shape
    h_co, w_co = coronal.shape
    h_sa, w_sa = sagittal.shape
    H = max(h_ax, h_co, h_sa)
    W = max(w_ax, w_co, w_sa)

    def pad_to(img, H, W):
        padded = np.zeros((H, W), dtype=img.dtype)
        padded[: img.shape[0], : img.shape[1]] = img
        return padded

    axial_pad = pad_to(axial, H, W)
    coronal_pad = pad_to(coronal, H, W)
    sagittal_pad = pad_to(sagittal, H, W)
    return axial_pad, coronal_pad, sagittal_pad

# =============================================================================
#                    Custom 3‑D Volume Transforms
# =============================================================================

class Resize3D:
    """Resize spatial dimensions (H and W) in a 3‑D volume."""

    def __init__(self, size):
        self.size = size  # (target_H, target_W)

    def __call__(self, volume):  # volume: (C, D, H, W)
        volume = volume.unsqueeze(0)  # (1, C, D, H, W)
        target_size = (volume.shape[2], self.size[0], self.size[1])
        volume = F.interpolate(volume, size=target_size, mode="trilinear", align_corners=False)
        return volume.squeeze(0)


class RandomFlip3D:
    """Random flips along spatial dims of a 3‑D volume."""

    def __init__(self, prob=0.5, dims=(2, 3)):
        self.prob = prob
        self.dims = dims

    def __call__(self, volume):
        if random.random() < self.prob:
            for d in self.dims:
                volume = torch.flip(volume, dims=[d])
        return volume


class Normalize3D:
    """Channel‑wise normalization for 3‑D volumes."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, volume):
        mean_t = torch.tensor(self.mean, dtype=volume.dtype, device=volume.device).view(-1, 1, 1, 1)
        std_t = torch.tensor(self.std, dtype=volume.dtype, device=volume.device).view(-1, 1, 1, 1)
        return (volume - mean_t) / std_t


class MedicalVolumeAugmentations3D:
    """Convenience wrapper around the transform pipelines."""

    def __init__(self, img_size: int, intensity: float = 0.5):
        self.train_transform = transforms.Compose(
            [
                RandomFlip3D(prob=intensity, dims=(2, 3)),
                Resize3D((img_size, img_size)),
                Normalize3D(
                    mean=[0.43216, 0.394666, 0.37645],
                    std=[0.22803, 0.22145, 0.216989],
                ),
            ]
        )
        self.val_transform = transforms.Compose(
            [
                Resize3D((img_size, img_size)),
                Normalize3D(
                    mean=[0.43216, 0.394666, 0.37645],
                    std=[0.22803, 0.22145, 0.216989],
                ),
            ]
        )

    def get(self, train: bool = True):
        return self.train_transform if train else self.val_transform


# ----------------------------------------------------------------------------
#                          Dataset definition
# ----------------------------------------------------------------------------

class MedicalImageStackDataset(Dataset):
    """Dataset that loads volumes saved as .npy."""

    def __init__(self, data_dir, split='train', transform=None, stack_method='multi', max_slices=64):
        self.transform = transform
        self.stack_method = stack_method
        self.max_slices = max_slices
        self.samples = []  # list[(file_path, label)]
        self.class_weights = {}

        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        class_counts = {}
        for class_name in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_name)
            # Special case for test/1/i folders
            if split == 'test' and class_name == '1' and os.path.isdir(os.path.join(class_path, 'i')):
                class_path = os.path.join(class_path, 'i')

            if os.path.isdir(class_path):
                try:
                    label = int(class_name)
                except ValueError:
                    continue  # skip non‑numeric folders

                npy_files = [f for f in os.listdir(class_path) if f.endswith('.npy')]
                self.samples.extend([(os.path.join(class_path, f), label) for f in npy_files])
                class_counts[label] = class_counts.get(label, 0) + len(npy_files)

        if class_counts:
            total = sum(class_counts.values())
            for lbl, cnt in class_counts.items():
                self.class_weights[lbl] = total / (len(class_counts) * cnt)

        logger.info(f"Loaded {len(self.samples)} samples from split='{split}'. Class dist: {class_counts}")

    # ---------- PyTorch mandatory ----------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        try:
            vol = np.load(file_path)  # (H, W, D)
            if vol.ndim != 3:
                raise ValueError(f"Expected 3‑D array, got {vol.shape}")
        except Exception as e:
            logger.error(f"Failed loading {file_path}: {e}")
            return torch.zeros((3, self.max_slices, IMG_SIZE, IMG_SIZE)), label

        vol = vol.transpose(2, 0, 1)  # (D, H, W)
        depth = vol.shape[0]
        if depth > self.max_slices:
            start = (depth - self.max_slices) // 2
            vol = vol[start: start + self.max_slices]
        elif depth < self.max_slices:
            pad = ((0, self.max_slices - depth), (0, 0), (0, 0))
            vol = np.pad(vol, pad, mode='constant')

        vol = torch.from_numpy(vol).float().unsqueeze(0)  # (1, D, H, W)
        vol = vol.repeat(3, 1, 1, 1)  # (3, D, H, W)
        if self.transform:
            vol = self.transform(vol)
        return vol, label

# ----------------------------------------------------------------------------
#                        Model creation util
# ----------------------------------------------------------------------------

def create_model(num_classes: int):
    logger.info(f"Creating model {MODEL_NAME}")
    model = r3d_18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ----------------------------------------------------------------------------
#               Training + evaluation for one split
# ----------------------------------------------------------------------------

def train_evaluate(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    early_stopping_patience,
    output_dir,
    use_amp=False,
):
    best_val_f1 = 0.0
    no_improve = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_auc': [],
        'val_loss':   [], 'val_acc':   [], 'val_f1':   [], 'val_auc':   []
    }

    scaler = GradScaler() if use_amp else None

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        # ---- TRAIN ----
        model.train()
        tr_loss = 0.0
        tr_preds, tr_labels, tr_probs = [], [], []
        for x, y in tqdm(train_loader, desc='train'):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if use_amp:
                with autocast():
                    out = model(x)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

            tr_loss += loss.item() * x.size(0)
            probs = torch.softmax(out, 1)
            _, preds = torch.max(out, 1)
            tr_labels.extend(y.cpu())
            tr_preds.extend(preds.cpu())
            tr_probs.extend(probs.detach().cpu())

        tr_loss /= len(train_loader.dataset)
        tr_acc = accuracy_score(tr_labels, tr_preds)
        _, _, tr_f1, _ = precision_recall_fscore_support(
            tr_labels, tr_preds, average='weighted', zero_division=0
        )
        tr_auc = _compute_auc(tr_labels, tr_probs)

        # ---- VAL ----
        model.eval()
        v_loss = 0.0
        v_preds, v_labels, v_probs = [], [], []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc='val'):
                x, y = x.to(device), y.to(device)
                if use_amp:
                    with autocast():
                        out = model(x)
                        loss = criterion(out, y)
                else:
                    out = model(x)
                    loss = criterion(out, y)
                v_loss += loss.item() * x.size(0)
                probs = torch.softmax(out, 1)
                _, preds = torch.max(out, 1)
                v_labels.extend(y.cpu())
                v_preds.extend(preds.cpu())
                v_probs.extend(probs.cpu())

        v_loss /= len(val_loader.dataset)
        v_acc = accuracy_score(v_labels, v_preds)
        _, _, v_f1, _ = precision_recall_fscore_support(
            v_labels, v_preds, average='weighted', zero_division=0
        )
        v_auc = _compute_auc(v_labels, v_probs)

        scheduler.step(v_f1)

        # log & save
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['train_f1'].append(tr_f1)
        history['train_auc'].append(tr_auc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        history['val_f1'].append(v_f1)
        history['val_auc'].append(v_auc)

        logger.info(
            f"tr_loss={tr_loss:.4f} tr_f1={tr_f1:.4f} | val_loss={v_loss:.4f} val_f1={v_f1:.4f}"
        )

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
        else:
            no_improve += 1
            logger.info(f'No improvement for {no_improve} epochs')
            if no_improve >= early_stopping_patience:
                logger.info('Early stopping!')
                break

    best_epoch = int(np.argmax(history['val_f1']))
    best_metrics = {
        'best_epoch': best_epoch + 1,
        'best_val_loss': history['val_loss'][best_epoch],
        'best_val_acc': history['val_acc'][best_epoch],
        'best_val_f1': history['val_f1'][best_epoch],
        'best_val_auc': history['val_auc'][best_epoch],
    }
    pd.DataFrame(history).to_csv(os.path.join(output_dir, 'history.csv'), index=False)
    return best_metrics


def _compute_auc(labels, probs):
    labels = np.array(labels)
    probs = np.vstack(probs)
    if probs.shape[1] == 2:
        return roc_auc_score(labels, probs[:, 1])
    else:
        return roc_auc_score(np.eye(probs.shape[1])[labels], probs, multi_class='ovr', average='weighted')

# ----------------------------------------------------------------------------
#                        Plot helpers (optional)
# ----------------------------------------------------------------------------

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ----------------------------------------------------------------------------
#                    Leave‑One‑Out Cross‑Validation
# ----------------------------------------------------------------------------

class SubsetWithTransform(Dataset):
    """Aplica um transform específico a um torch.utils.data.Subset."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]        # volume ainda cru
        if self.transform is not None: # transforma aqui
            x = self.transform(x)
        return x, y
# ---------------------------------------------------------------------------


def run_leave_one_out():
    """
    Leave‑One‑Out:
      • Usa 100 % dos dados disponíveis (train + test) num único pool
      • ✅ NÃO aplica augmentação à amostra de validação
      • ✅ Aplica augmentação apenas às amostras de treino
    """
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_out = os.path.join(OUTPUT_DIR, f"LOO_{MODEL_NAME}_{timestamp}")
    os.makedirs(base_out, exist_ok=True)

    # Pipelines de transformação
    aug = MedicalVolumeAugmentations3D(IMG_SIZE, AUG_INTENSITY)
    train_tf = aug.get(True)   # com RandomFlip3D
    val_tf   = aug.get(False)  # só Resize3D + Normalize3D

    # Conjunto completo SEM transformações (serão aplicadas via SubsetWithTransform)
    train_raw = MedicalImageStackDataset(
        DATA_DIR, split='train', transform=None,
        stack_method=STACK_METHOD, max_slices=MAX_SLICES
    )
    test_raw = MedicalImageStackDataset(
        DATA_DIR, split='test',  transform=None,
        stack_method=STACK_METHOD, max_slices=MAX_SLICES
    )
    full_dataset = ConcatDataset([train_raw, test_raw])
    logger.info(f"Dataset completo para LOO: {len(full_dataset)} amostras")

    # Recalcula pesos de classe (com ambos splits)
    from collections import Counter
    counts = Counter(lbl for ds in [train_raw, test_raw] for _, lbl in ds.samples)
    total = sum(counts.values())
    class_weights = {lbl: total / (len(counts) * cnt) for lbl, cnt in counts.items()}

    overall_metrics = []
    for idx in range(len(full_dataset)):
        logger.info(f"\n========== LOO {idx + 1}/{len(full_dataset)} ==========")

        # Separam treino / validação (sem transform ainda)
        val_raw_subset   = Subset(full_dataset, [idx])
        train_raw_subset = Subset(full_dataset, [i for i in range(len(full_dataset)) if i != idx])

        # Aplica transform específico para cada split
        train_subset = SubsetWithTransform(train_raw_subset, train_tf)
        val_subset   = SubsetWithTransform(val_raw_subset,   val_tf)

        train_loader = DataLoader(
            train_subset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=1, pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_subset, batch_size=1, shuffle=False,
            num_workers=1, pin_memory=torch.cuda.is_available()
        )

        model = create_model(NUM_CLASSES).to(device)
        weights = [class_weights.get(i, 1.0) for i in range(NUM_CLASSES)]
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).float().to(device))
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

        iter_out = os.path.join(base_out, f'fold_{idx:03d}')
        os.makedirs(iter_out, exist_ok=True)
        best_metrics = train_evaluate(
            model, train_loader, val_loader,
            criterion, optimizer, scheduler,
            device, EPOCHS, EARLY_STOPPING,
            iter_out, USE_AMP,
        )
        overall_metrics.append(best_metrics)

        # Matriz de confusão do fold
        model.eval()
        x_val, y_val = val_subset[0]
        with torch.no_grad():
            pred = torch.argmax(model(x_val.unsqueeze(0).to(device)), 1).item()
        cm = confusion_matrix([y_val], [pred], labels=list(range(NUM_CLASSES)))
        plot_confusion_matrix(
            cm, [str(c) for c in range(NUM_CLASSES)],
            os.path.join(iter_out, 'cm.png')
        )

    # Resumo geral
    df = pd.DataFrame(overall_metrics)
    df.loc['mean'] = df.mean()
    df.to_csv(os.path.join(base_out, 'loo_summary.csv'), index=False)
    logger.info("LOO concluído. Resumo:\n" + df.tail(1).to_string(index=False))

# ----------------------------------------------------------------------------
#                                   main()
# ----------------------------------------------------------------------------

def standard_train():
    """Original training (train vs test split)."""
    set_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # log to file
    fh = logging.FileHandler(os.path.join(out_dir, 'train.log'))
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(fh)
    logger.info(f'Config: {param_config}')

    aug = MedicalVolumeAugmentations3D(IMG_SIZE, AUG_INTENSITY)
    train_ds = MedicalImageStackDataset(DATA_DIR, 'train', aug.get(True), STACK_METHOD, MAX_SLICES)
    val_ds = MedicalImageStackDataset(DATA_DIR, 'test', aug.get(False), STACK_METHOD, MAX_SLICES)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=torch.cuda.is_available())

    model = create_model(NUM_CLASSES).to(device)
    weights = [train_ds.class_weights.get(i, 1.0) for i in range(NUM_CLASSES)]
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).float().to(device))
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

    best = train_evaluate(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        EPOCHS,
        EARLY_STOPPING,
        out_dir,
        USE_AMP,
    )
    logger.info(f'Best metrics: {best}')


if __name__ == '__main__':
    if RUN_LEAVE_ONE_OUT:
        run_leave_one_out()
    else:
        standard_train()