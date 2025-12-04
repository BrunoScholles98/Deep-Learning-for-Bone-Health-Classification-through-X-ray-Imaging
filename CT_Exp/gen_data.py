# -*- coding: utf-8 -*-
"""
3-D Data Preparation - Modes 1-5
===================================
Mode 5 creates class-based folders in train/test, with a patient subfolder
inside each class containing three .npy files (Axial, Coronal, Sagittal)
with 64 centered slices.

Example structure:
output_folder/
└── train/
    ├── 0/
    │   ├── Paciente00012/
    │   │   ├── stack_00012_0_Axial.npy
    │   │   ├── stack_00012_0_Coronal.npy
    │   │   └── stack_00012_0_Sagital.npy
    │   └── …
    └── 1/
        ├── Paciente00034/
        │   └── …
        └── (test class 1 may have an 'i' subfolder if needed)
"""
# Imports
import os
import nrrd
import numpy as np
import pandas as pd
import random
import math
from tqdm import tqdm

# Helper functions
def normalize_to_255(array_slice: np.ndarray) -> np.ndarray:
    """Normalize a 2D slice to uint8 in [0, 255]."""
    arr = array_slice.astype(np.float32)
    min_val, max_val = arr.min(), arr.max()
    if max_val > min_val:
        arr = (arr - min_val) / (max_val - min_val)
    else:
        arr = arr - min_val
    arr *= 255.0
    return arr.astype(np.uint8)


def get_3_views(im_data: np.ndarray, cx: int, cy: int, cz: int):
    """Returns normalized axial, coronal, and sagittal slices crossing (cx, cy, cz)."""
    axial    = normalize_to_255(im_data[:, :, cz])
    coronal  = normalize_to_255(im_data[:, cy, :])
    sagittal = normalize_to_255(im_data[cx, :, :])
    return axial, coronal, sagittal


# Configuration
# 1 = segmentations; 2 = random; 3 = center of segmentation
# 4 = single stack (view); 5 = three views, 64 slices each
data_collection_mode = 5

selected_view    = "3_Views"  # only affects mode 4
stack_size_mode4 = 64         # mode 4
stack_size_mode5 = 64         # mode 5 (fixed)

excel_ref      = r"C:\Gigasistemica\Datasets\3D\Dados_clínicos_densitométricos_pacientes.xlsx"
col_ref_output = "Diagnosticounificado"
col_number     = "Número da TC"

folder_imgs   = r"C:\Gigasistemica\Datasets\3D\Scene3DSlicer_avaliacao_Lorena_osteoporose"
output_folder = rf"TM_3D_64Stacks_{selected_view}"

train_split   = 0.8  # 80% train | 20% test

# Prepare base directories
df = pd.read_excel(excel_ref)

os.makedirs(output_folder, exist_ok=True)
for split in ("train", "test"):
    os.makedirs(os.path.join(output_folder, split), exist_ok=True)

patients = [p for p in os.listdir(folder_imgs) if p.startswith("Paciente")]
random.shuffle(patients)

train_size     = math.floor(len(patients) * train_split)
train_patients = set(patients[:train_size])
test_patients  = set(patients[train_size:])

# Main loop
for patient_folder_name in tqdm(patients, desc="Processing patients"):
    patient_id = patient_folder_name.replace("Paciente", "")

    try:
        patient_id_lookup = int(patient_id)
    except ValueError:
        patient_id_lookup = patient_id

    split = "train" if patient_folder_name in train_patients else "test"

    # Get label/class
    row = df[df[col_number] == patient_id_lookup]
    if row.empty:
        print(f"► No Excel info for {patient_id}, skipping…")
        continue

    try:
        diagnostic_label = int(row.iloc[0][col_ref_output])
    except ValueError:
        diagnostic_label = str(row.iloc[0][col_ref_output]).strip()
    diagnostic_label_str = str(diagnostic_label)

    # File paths
    patient_folder = os.path.join(folder_imgs, patient_folder_name)
    seg_path = os.path.join(patient_folder, "Segmentation.seg.nrrd")
    img_path = os.path.join(patient_folder, "0 Unnamed Series.nrrd")

    # Check file existence
    if data_collection_mode == 1 and not (os.path.exists(seg_path) and os.path.exists(img_path)):
        print(f"► Missing files (mode 1) for {patient_id}, skipping…"); continue
    elif data_collection_mode == 2 and not os.path.exists(img_path):
        print(f"► No image (mode 2) for {patient_id}, skipping…"); continue
    elif data_collection_mode == 3 and not (os.path.exists(seg_path) and os.path.exists(img_path)):
        print(f"► No seg/img (mode 3) for {patient_id}, skipping…"); continue
    elif data_collection_mode == 4 and not os.path.exists(img_path):
        print(f"► No image (mode 4) for {patient_id}, skipping…"); continue
    elif data_collection_mode == 5 and not os.path.exists(img_path):
        print(f"► No image (mode 5) for {patient_id}, skipping…"); continue

    # -----------------------------------------------------------
    # MODE 1 - COMPLETE SEGMENTATIONS
    # -----------------------------------------------------------
    if data_collection_mode == 1:
        seg_data, _ = nrrd.read(seg_path)
        im_data, _  = nrrd.read(img_path)

        unique_seg_values = np.unique(seg_data)
        unique_seg_values = unique_seg_values[unique_seg_values > 0]
        if len(unique_seg_values) == 0:
            print(f"► No segmentation in {patient_id}, skipping…")
            continue

        label_dir = os.path.join(output_folder, split, diagnostic_label_str)
        if split == "test" and diagnostic_label_str == "1":
            label_dir = os.path.join(label_dir, "i")
        os.makedirs(label_dir, exist_ok=True)

        for seg_value in unique_seg_values:
            seg_value_int = int(seg_value)
            slices_idx = [i for i in range(seg_data.shape[-1])
                          if np.any(seg_data[:, :, i] == seg_value)]
            if not slices_idx:
                continue

            stack = [normalize_to_255(im_data[:, :, idx]) for idx in slices_idx]
            stack_array = np.stack(stack, axis=-1)

            fname = f"stack_{patient_id}_{diagnostic_label_str}_{seg_value_int}.npy"
            np.save(os.path.join(label_dir, fname), stack_array)
            print(f"[Mode 1] seg={seg_value_int} saved → {patient_id} ({split})")

    # -----------------------------------------------------------
    # MODE 2 - RANDOM STACKS
    # -----------------------------------------------------------
    elif data_collection_mode == 2:
        im_data, _ = nrrd.read(img_path)
        total_slices = im_data.shape[-1]

        if total_slices < 10:
            print(f"► Too few slices in {patient_id}, skipping…")
            continue

        lower_bound = int(total_slices * 0.1) + 4
        upper_bound = int(total_slices * 0.9) - 4
        if lower_bound > upper_bound:
            print(f"► Invalid range in {patient_id}, skipping…")
            continue

        label_dir = os.path.join(output_folder, split, diagnostic_label_str)
        if split == "test" and diagnostic_label_str == "1":
            label_dir = os.path.join(label_dir, "i")
        os.makedirs(label_dir, exist_ok=True)

        for stack_num in range(5):
            central_idx = random.randint(lower_bound, upper_bound)
            slice_indices = [central_idx - 4, central_idx, central_idx + 4]

            slices = [normalize_to_255(im_data[:, :, idx]) for idx in slice_indices]
            stack_array = np.stack(slices, axis=-1)

            fname = f"stack_{patient_id}_{diagnostic_label_str}_{stack_num}.npy"
            np.save(os.path.join(label_dir, fname), stack_array)
            print(f"[Mode 2] stack={stack_num} saved → {patient_id} ({split})")

    # -----------------------------------------------------------
    # MODE 3 - 3 VIEWS AT SEGMENTATION CENTER
    # -----------------------------------------------------------
    elif data_collection_mode == 3:
        seg_data, _ = nrrd.read(seg_path)
        im_data, _  = nrrd.read(img_path)

        unique_seg_values = np.unique(seg_data)
        unique_seg_values = unique_seg_values[unique_seg_values > 0]
        if len(unique_seg_values) == 0:
            print(f"► No segmentation in {patient_id}, skipping…")
            continue

        label_dir = os.path.join(output_folder, split, diagnostic_label_str)
        if split == "test" and diagnostic_label_str == "1":
            label_dir = os.path.join(label_dir, "i")
        os.makedirs(label_dir, exist_ok=True)

        dimX, dimY, dimZ = seg_data.shape
        for seg_value in unique_seg_values:
            coords = np.argwhere(seg_data == seg_value)
            if coords.size == 0:
                continue

            cx, cy, cz = np.round(coords.mean(axis=0)).astype(int)
            if not (0 <= cx < dimX and 0 <= cy < dimY and 0 <= cz < dimZ):
                continue

            axial, coronal, sagittal = get_3_views(im_data, cx, cy, cz)
            fname = f"stack_{patient_id}_{diagnostic_label_str}_{int(seg_value)}_views.npz"
            np.savez(os.path.join(label_dir, fname),
                     axial=axial, coronal=coronal, sagittal=sagittal)
            print(f"[Mode 3] seg={int(seg_value)} saved → {patient_id} ({split})")

    # -----------------------------------------------------------
    # MODE 4 - SINGLE STACK IN SELECTED VIEW
    # -----------------------------------------------------------
    elif data_collection_mode == 4:
        im_data, _ = nrrd.read(img_path)

        label_dir = os.path.join(output_folder, split, diagnostic_label_str)
        if split == "test" and diagnostic_label_str == "1":
            label_dir = os.path.join(label_dir, "i")
        os.makedirs(label_dir, exist_ok=True)

        if selected_view == "Axial":
            axis = 2; extractor = lambda idx: im_data[:, :, idx]
        elif selected_view == "Coronal":
            axis = 1; extractor = lambda idx: im_data[:, idx, :]
        elif selected_view == "Sagital":
            axis = 0; extractor = lambda idx: im_data[idx, :, :]
        else:
            print(f"► Invalid view ({selected_view}) in {patient_id}, skipping…")
            continue

        total = im_data.shape[axis]
        central = total // 2
        lower = stack_size_mode4 // 2
        upper = stack_size_mode4 - lower
        start = central - lower
        end   = central + upper

        if start < 0 or end > total:
            print(f"► Indices out of range in {patient_id}, skipping…")
            continue

        stack = [normalize_to_255(extractor(idx)) for idx in range(start, end)]
        stack_array = np.stack(stack, axis=-1)

        fname = f"stack_{patient_id}_{diagnostic_label_str}_mode4_{selected_view}.npy"
        np.save(os.path.join(label_dir, fname), stack_array)
        print(f"[Mode 4] {selected_view:7} saved → {patient_id} ({split})")

    # -----------------------------------------------------------
    # MODE 5 - 64 SLICES IN 3 VIEWS (CORRECTED)
    # -----------------------------------------------------------
    elif data_collection_mode == 5:
        im_data, _ = nrrd.read(img_path)

        views = {
            "Axial":   (2, lambda idx: im_data[:, :, idx]),
            "Coronal": (1, lambda idx: im_data[:, idx, :]),
            "Sagital": (0, lambda idx: im_data[idx, :, :]),
        }

        # Create class folder
        label_dir = os.path.join(output_folder, split, diagnostic_label_str)
        if split == "test" and diagnostic_label_str == "1":
            label_dir = os.path.join(label_dir, "i")
        os.makedirs(label_dir, exist_ok=True)

        # Create patient folder inside class
        patient_dir = os.path.join(label_dir, f"Paciente{patient_id}")
        os.makedirs(patient_dir, exist_ok=True)

        for view_name, (axis, extractor) in views.items():
            total = im_data.shape[axis]
            if total < stack_size_mode5:
                print(f"► {view_name}: too few slices in {patient_id}, skipping view.")
                continue

            central = total // 2
            lower   = stack_size_mode5 // 2
            upper   = stack_size_mode5 - lower
            start   = central - lower
            end     = central + upper

            if start < 0 or end > total:
                print(f"► {view_name}: indices out of range in {patient_id}.")
                continue

            stack = [normalize_to_255(extractor(idx)) for idx in range(start, end)]
            stack_array = np.stack(stack, axis=-1)

            fname = f"stack_{patient_id}_{diagnostic_label_str}_{view_name}.npy"
            np.save(os.path.join(patient_dir, fname), stack_array)
            print(f"[Mode 5] {view_name:7} saved → {patient_id} ({split})")

print(">>> Processing complete! ✅")