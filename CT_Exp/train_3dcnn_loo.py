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
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import logging

GPU_IDS           = [0]
DATA_DIR          = '/mnt/nas/BrunoScholles/Gigasistemica/Datasets/TM_3D_64Stacks_Axial'
OUTPUT_DIR        = '/mnt/nas/BrunoScholles/Gigasistemica/Models'
MODEL_NAME        = 'r3d_18'

NUM_CLASSES       = 2
EPOCHS            = 50
BATCH_SIZE        = 16
LR                = 1e-4
WEIGHT_DECAY      = 1e-4
SEED              = 42
IMG_SIZE          = 112
MAX_SLICES        = 64
AUG_INTENSITY     = 0.99   # used for RandomFlip3D
N_AUG_PER_SAMPLE  = 1      # how many augmented copies per sample at each LOO
USE_AMP           = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """Controls all sources of randomness (CPU and GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 3D Transforms
class Resize3D:
    def __init__(self, size):
        self.size = size  # (H, W)
    def __call__(self, vol):
        # vol: tensor (C=3, D, H, W)
        # we want to resize HxW, keeping D
        vol = vol.unsqueeze(0)  # (1,3,D,H,W)
        vol = F.interpolate(
            vol,
            size=(vol.shape[2], *self.size),
            mode="trilinear",
            align_corners=False
        )  # (1,3,D,new_H,new_W)
        return vol.squeeze(0)   # (3, D, new_H, new_W)

class RandomFlip3D:
    def __init__(self, prob=0.5, dims=(2, 3)):
        self.prob = prob
        self.dims = dims
    def __call__(self, vol):
        # vol: tensor (3, D, H, W)
        if random.random() < self.prob:
            for d in self.dims:
                vol = torch.flip(vol, dims=[d])
        return vol

class Normalize3D:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, vol):
        # vol: tensor (3, D, H, W)
        mean_t = torch.tensor(self.mean, dtype=vol.dtype, device=vol.device).view(-1,1,1,1)
        std_t  = torch.tensor(self.std,  dtype=vol.dtype, device=vol.device).view(-1,1,1,1)
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

# Raw Dataset (loads .npy, without transform)
class MedicalImageStackDataset(Dataset):
    def __init__(self, data_dir, split, transform, max_slices=64):
        self.transform = transform
        self.max_slices = max_slices
        self.samples = []

        split_dir = os.path.join(data_dir, split)
        for class_name in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_name)
            # Specific adjustment for 'test/1/i' if it exists
            if split == 'test' and class_name == '1' and os.path.isdir(os.path.join(class_path, 'i')):
                class_path = os.path.join(class_path, 'i')
            if not os.path.isdir(class_path):
                continue
            label = int(class_name)
            for f in os.listdir(class_path):
                if f.endswith('.npy'):
                    self.samples.append((os.path.join(class_path, f), label))
        logger.info(f'Loaded {split}: {len(self.samples)} samples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        vol = np.load(path).transpose(2,0,1)  # (D, H, W)
        d = vol.shape[0]
        # Truncate or pad to have D = MAX_SLICES
        if d > self.max_slices:
            s = (d - self.max_slices) // 2
            vol = vol[s : s + self.max_slices]
        elif d < self.max_slices:
            vol = np.pad(vol, ((0, self.max_slices - d), (0,0), (0,0)))
        # From (D, H, W) -> (1, D, H, W) -> (3, D, H, W)
        vol_tensor = torch.from_numpy(vol).float().unsqueeze(0).repeat(3,1,1,1)
        if self.transform:
            vol_tensor = self.transform(vol_tensor)
        return vol_tensor, label

# Dataset that loads only original volumes from split (no augmentation),
# applying resize+normalization via passed transform
class OrigTrainDataset(Dataset):
    def __init__(self, samples, transform, max_slices=64):
        """
        samples: list of tuples (path.npy, label)
        transform: only Resize3D + Normalize3D (no flip)
        """
        self.samples = samples
        self.transform = transform
        self.max_slices = max_slices

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        vol = np.load(path).transpose(2,0,1)  # (D, H, W)
        d = vol.shape[0]
        if d > self.max_slices:
            s = (d - self.max_slices) // 2
            vol = vol[s : s + self.max_slices]
        elif d < self.max_slices:
            vol = np.pad(vol, ((0, self.max_slices - d), (0,0), (0,0)))
        vol_tensor = torch.from_numpy(vol).float().unsqueeze(0).repeat(3,1,1,1)
        vol_tensor = self.transform(vol_tensor)
        return vol_tensor, label

# Dataset for volumes already loaded in memory (augmented)
class InMemoryDataset(Dataset):
    def __init__(self, vol_list, label_list):
        """
        vol_list: list of tensors (3, D, H, W), already normalized
        label_list: list of integer labels
        """
        assert len(vol_list) == len(label_list)
        self.vols = vol_list
        self.labels = label_list

    def __len__(self):
        return len(self.vols)

    def __getitem__(self, idx):
        return self.vols[idx], self.labels[idx]

# R3D-18 Model
def create_model(num_classes):
    m = r3d_18(weights=R3D_18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

# Training function per split (doesn't change)
def train_one_split(model, train_loader, criterion, optimizer,
                    device, epochs, use_amp=False, loo_idx=None, total_loo=None):
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for ep in range(epochs):
        if loo_idx is not None and total_loo is not None:
            logger.info(f'  LOO {loo_idx}/{total_loo} - Epoch {ep+1}/{epochs}')
        else:
            logger.info(f'  Epoch {ep+1}/{epochs}')
        model.train()
        for x, y in tqdm(train_loader, leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
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

    return model

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Leave-One-Out (LOO) pipeline with generation of new volumes at each iteration
def run_leave_one_out():
    set_seed(SEED)

    # Select device
    if torch.cuda.is_available() and len(GPU_IDS) > 0:
        device = torch.device(f'cuda:{GPU_IDS[0]}')
        logger.info(f'Using GPU(s): {GPU_IDS}')
    else:
        device = torch.device('cpu')
        logger.info('CUDA not available – using CPU')

    # Create output folder
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(OUTPUT_DIR, f'LOO_{MODEL_NAME}_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    # Instantiate augmentations (train_tf includes flip + resize + normalize; val_tf only resize+normalize)
    aug = MedicalVolumeAugmentations3D(IMG_SIZE, AUG_INTENSITY)

    # Load raw train and test samples (without transform)
    train_raw_full = MedicalImageStackDataset(DATA_DIR, 'train', None, MAX_SLICES)
    test_raw_full  = MedicalImageStackDataset(DATA_DIR, 'test',  None, MAX_SLICES)

    # Concatenate sample lists [(path,label), ...]
    full_samples = train_raw_full.samples + test_raw_full.samples
    total = len(full_samples)
    logger.info(f'LOO over {total} samples.')

    # Compute class weights based on total count
    counts = Counter(lbl for _, lbl in full_samples)
    tot = sum(counts.values())
    cls_w = torch.tensor([tot / (len(counts) * counts[i]) for i in range(NUM_CLASSES)])

    global_true, global_pred, global_prob = [], [], []

    # LOO loop
    for idx in range(total):
        logger.info(f'\n========== LOO {idx+1}/{total} ==========')

        # Separate validation sample
        val_path, val_label = full_samples[idx]
        # Build list of training samples (all except idx)
        train_samples = [full_samples[i] for i in range(total) if i != idx]

        orig_train_ds = OrigTrainDataset(
            samples=train_samples,
            transform=aug.get(False),
            max_slices=MAX_SLICES
        )

        aug_vols = []
        aug_labels = []
        for (path, label) in train_samples:
            # Load raw volume
            vol_np = np.load(path).transpose(2,0,1)  # (D, H, W)
            d = vol_np.shape[0]
            # Truncate or pad to have D = MAX_SLICES
            if d > MAX_SLICES:
                s = (d - MAX_SLICES) // 2
                vol_np = vol_np[s : s + MAX_SLICES]
            elif d < MAX_SLICES:
                vol_np = np.pad(vol_np, ((0, MAX_SLICES - d), (0,0), (0,0)))
            # Convert to tensor (3, D, H, W)
            vol_tensor = torch.from_numpy(vol_np).float().unsqueeze(0).repeat(3,1,1,1)
            # For each desired augmented copy:
            for _ in range(N_AUG_PER_SAMPLE):
                # Apply flip + resize + normalize
                vol_aug = aug.get(True)(vol_tensor)  # (3, D, IMG_SIZE, IMG_SIZE), normalized
                aug_vols.append(vol_aug)
                aug_labels.append(label)

        aug_ds = InMemoryDataset(aug_vols, aug_labels)

        # Concatenate original + augmented dataset
        train_ds = ConcatDataset([orig_train_ds, aug_ds])

        train_ld = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=torch.cuda.is_available()
        )

        # Create model
        model = create_model(NUM_CLASSES)
        if torch.cuda.is_available() and len(GPU_IDS) > 1:
            model = nn.DataParallel(model, device_ids=GPU_IDS)
        model = model.to(device)

        crit = nn.CrossEntropyLoss(weight=cls_w.to(device))
        opt  = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        # Train on current split (orig + aug volumes)
        model = train_one_split(
            model,
            train_ld,
            crit,
            opt,
            device,
            EPOCHS,
            USE_AMP,
            loo_idx=idx+1,
            total_loo=total
        )

        # Load validation volume
        vol_np = np.load(val_path).transpose(2,0,1)
        d = vol_np.shape[0]
        if d > MAX_SLICES:
            s = (d - MAX_SLICES) // 2
            vol_np = vol_np[s : s + MAX_SLICES]
        elif d < MAX_SLICES:
            vol_np = np.pad(vol_np, ((0, MAX_SLICES - d), (0,0), (0,0)))
        vol_tensor = torch.from_numpy(vol_np).float().unsqueeze(0).repeat(3,1,1,1)
        vol_val = aug.get(False)(vol_tensor).to(device)  # resize+normalize
        y_val = val_label

        model.eval()
        with torch.no_grad():
            out = model(vol_val.unsqueeze(0))  # add batch dimension
            prob = torch.softmax(out, dim=1).cpu().squeeze(0)
        pred = prob.argmax().item()

        global_true.append(y_val)
        global_pred.append(pred)
        global_prob.append(prob.numpy())

    cm = confusion_matrix(global_true, global_pred, labels=[0, 1])
    plot_confusion_matrix(
        cm,
        class_names=[str(i) for i in range(NUM_CLASSES)],
        save_path=os.path.join(out_dir, 'confusion_matrix_overall.png')
    )

    tn, fp, fn, tp = cm.ravel()
    accuracy    = accuracy_score(global_true, global_pred)
    precision   = precision_score(global_true, global_pred, zero_division=0)
    recall      = recall_score(global_true, global_pred, zero_division=0)
    f1          = f1_score(global_true, global_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    npv         = tn / (tn + fn) if (tn + fn) else 0.0
    auc         = roc_auc_score(global_true, np.vstack(global_prob)[:, 1])

    # Save metrics to file
    with open(os.path.join(out_dir, 'loo_metrics.txt'), 'w') as f:
        f.write(f'Accuracy   : {accuracy:.4f}\n')
        f.write(f'Precision  : {precision:.4f}\n')
        f.write(f'Recall     : {recall:.4f}\n')
        f.write(f'Specificity: {specificity:.4f}\n')
        f.write(f'NPV        : {npv:.4f}\n')
        f.write(f'F1-Score   : {f1:.4f}\n')
        f.write(f'AUC        : {auc:.4f}\n')

    logger.info('\n=== LOO Performance ===')
    logger.info(open(os.path.join(out_dir, 'loo_metrics.txt')).read().strip())

if __name__ == '__main__':
    run_leave_one_out()