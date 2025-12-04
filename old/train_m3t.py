#!/usr/bin/env python
"""
Treinamento do M3T — 3 stacks por paciente
==========================================
Cada amostra  ➜  tensor (3, D, H, W)
  canal 0: Axial   | 1: Coronal | 2: Sagital
Todas as variáveis de configuração são globais.
"""

# Configuration
DATA_DIR      = "/mnt/ssd/brunoscholles/GigaSistemica/Datasets/TM_3D_64Stacks_3_Views"
OUTPUT_ROOT   = "/mnt/ssd/brunoscholles/GigaSistemica/Models/TC_Models"

IMG_SIZE      = 112
BATCH_SIZE    = 1
EPOCHS        = 40
LR            = 3e-4
WEIGHT_DECAY  = 3e-5
EARLY_STOP    = 7
SEED          = 42
OUT_CHANNELS  = 16
NUM_CLASSES   = 2


import os, random, json, logging
from datetime import datetime
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from M3T import M3T

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO)
log = logging.getLogger("M3T-train")

# ----------------------------- utilidades -----------------------------------
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.benchmark = True

def auc_score(y_true, y_prob):
    """Retorna AUC ou np.nan se não houver pelo menos 2 classes."""
    y_true = np.array(y_true)
    y_prob = np.vstack(y_prob)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if y_prob.shape[1] == 2:
        return roc_auc_score(y_true, y_prob[:, 1])
    return roc_auc_score(np.eye(y_prob.shape[1])[y_true],
                         y_prob, multi_class="ovr", average="weighted")

def pad_to_shape(t, target_dhw):
    D,H,W = t.shape
    Td,Th,Tw = target_dhw
    return F.pad(t, (0,Tw-W, 0,Th-H, 0,Td-D))  # ordem: W,H,D

# ------------------------------ dataset -------------------------------------
class TripletVolumeDataset(Dataset):
    """Retorna tensor (3,D,H,W) após pad para forma comum."""
    def __init__(self, root, split, transform=None):
        self.transform = transform
        self.samples   = []
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(split_dir)

        for cls in sorted(os.listdir(split_dir)):
            cls_path = os.path.join(split_dir, cls)
            if not (os.path.isdir(cls_path) and cls.isdigit()):
                continue
            label = int(cls)
            for patient in os.listdir(cls_path):
                p_dir = os.path.join(cls_path, patient)
                if not os.path.isdir(p_dir):
                    continue
                paths = {}
                for f in os.listdir(p_dir):
                    if f.endswith(".npy"):
                        if "_Axial"   in f: paths["ax"] = os.path.join(p_dir, f)
                        if "_Coronal" in f: paths["co"] = os.path.join(p_dir, f)
                        if "_Sagital" in f: paths["sa"] = os.path.join(p_dir, f)
                if len(paths) == 3:
                    self.samples.append(((paths["ax"], paths["co"], paths["sa"]), label))

    def __len__(self): return len(self.samples)

    @staticmethod
    def _load(path):
        arr = np.load(path).astype(np.float32)           # (H,W,D)
        return torch.from_numpy(arr.transpose(2,0,1))    # (D,H,W)

    def __getitem__(self, idx):
        (p_ax,p_co,p_sa), lbl = self.samples[idx]
        vols = [self._load(p) for p in (p_ax,p_co,p_sa)]

        # pad p/ forma comum
        D = max(v.shape[0] for v in vols)
        H = max(v.shape[1] for v in vols)
        W = max(v.shape[2] for v in vols)
        vols = [pad_to_shape(v,(D,H,W)) for v in vols]

        vol = torch.stack(vols, 0)                       # (3,D,H,W)
        if self.transform: vol = self.transform(vol)
        return vol, lbl

# ----------------------------- transforms -----------------------------------
class Resize3D:
    def __init__(self,size): self.size=size
    def __call__(self,v):
        return F.interpolate(v.unsqueeze(0), (self.size,)*3,
                             mode="trilinear", align_corners=False).squeeze(0)

class RandomFlip3D:
    def __init__(self,p=0.5): self.p=p
    def __call__(self,v):
        if random.random()<self.p:
            v = torch.flip(v, dims=[random.choice([1,2,3])])
        return v

class Normalize3D:
    def __call__(self,v):
        mean = v.mean(dim=(1,2,3), keepdim=True)
        std  = v.std (dim=(1,2,3), keepdim=True) + 1e-5
        return (v-mean)/std

def get_tf(train=True):
    ops=[Resize3D(IMG_SIZE), Normalize3D()]
    if train: ops.insert(0, RandomFlip3D())
    return transforms.Compose(ops)

# ----------------------------- train/val ------------------------------------
def run_epoch(model, loader, criterion, optimizer, scaler, device, train=True):
    model.train() if train else model.eval()
    m = defaultdict(list)

    for x,y in tqdm(loader, desc="train" if train else "val", leave=False):
        x,y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            with autocast():
                out  = model(x)
                loss = criterion(out, y)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        probs = torch.softmax(out.detach(), 1)
        preds = probs.argmax(1)

        m["loss"].append(loss.item()*x.size(0))
        m["labels"].extend(y.cpu()); m["preds"].extend(preds.cpu())
        m["probs"].extend(probs.cpu())

    loss = sum(m["loss"])/len(loader.dataset)
    acc  = accuracy_score(m["labels"], m["preds"])
    _,_,f1,_ = precision_recall_fscore_support(m["labels"], m["preds"],
                                              average='weighted', zero_division=0)
    auc = auc_score(m["labels"], m["probs"])
    return loss, acc, f1, auc

# -------------------------------- main --------------------------------------
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    train_ds = TripletVolumeDataset(DATA_DIR, "train", get_tf(True))
    val_ds   = TripletVolumeDataset(DATA_DIR, "test",  get_tf(False))

    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

    model     = M3T(in_ch=3, out_ch=OUT_CHANNELS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler    = GradScaler()

    run_dir = os.path.join(OUTPUT_ROOT,
                           datetime.now().strftime("M3T_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    cfg = {k:v for k,v in globals().items()
           if k.isupper() and isinstance(v,(int,float,str,bool))}
    json.dump(cfg, open(os.path.join(run_dir,"config.json"),"w"))

    best_f1, patience = 0., 0
    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_acc, tr_f1, _ = run_epoch(model, train_ld, criterion,
                                             optimizer, scaler, device, True)
        vl_loss, vl_acc, vl_f1, _ = run_epoch(model, val_ld,   criterion,
                                             optimizer, scaler, device, False)

        log.info(f"[{epoch:02d}/{EPOCHS}] "
                 f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} tr_f1={tr_f1:.3f} | "
                 f"val_loss={vl_loss:.4f} val_acc={vl_acc:.3f} val_f1={vl_f1:.3f}")

        if vl_f1 > best_f1:
            best_f1, patience = vl_f1, 0
            torch.save(model.state_dict(), os.path.join(run_dir,"best.pth"))
        else:
            patience += 1
            if patience >= EARLY_STOP:
                log.info("Early stopping."); break

    log.info(f"Melhor F1 (val): {best_f1:.3f}")

if __name__ == "__main__":
    main()