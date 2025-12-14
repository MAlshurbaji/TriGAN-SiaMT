import os
import random
import numpy as np
import nibabel as nib
from PIL import Image
from skimage.transform import resize
from tqdm import tqdm
from nibabel.filebasedimages import ImageFileError

"""
3D → 2D Dataset Preparation Script
=================================
Supported datasets:
- ISLES 2022 (DWI + binary stroke mask)
- BraTS 2019 (FLAIR + binary tumor mask)

Outputs 2D PNG slices with patient-wise train/val/test split.
"""

# ============================================================
# CONFIGURATION
# ============================================================
TARGET_SHAPE = (73, 128, 128)  # (D, H, W)

TRAIN_RATIO = 0.75
VAL_RATIO   = 0.10
TEST_RATIO  = 0.15

SEED = 42


# ============================================================
# UTILITIES
# ============================================================
def normalize_to_uint8(volume: np.ndarray) -> np.ndarray:
    """Normalize volume to [0, 255] uint8."""
    vmin, vmax = volume.min(), volume.max()
    if vmax > vmin:
        volume = (volume - vmin) / (vmax - vmin)
    return (volume * 255).clip(0, 255).astype(np.uint8)


def split_patients(patient_ids):
    """Shuffle and split patient IDs."""
    random.shuffle(patient_ids)

    n_total = len(patient_ids)
    n_train = int(n_total * TRAIN_RATIO)
    n_val   = int(n_total * VAL_RATIO)

    return (
        patient_ids[:n_train],
        patient_ids[n_train:n_train + n_val],
        patient_ids[n_train + n_val:]
    )


def ensure_dirs(base_out):
    """Create output directories."""
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(base_out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(base_out, "labels", split), exist_ok=True)


def init_stats():
    """Initialize statistics dictionary."""
    return {
        "train": {"patients": 0, "slices": 0},
        "val":   {"patients": 0, "slices": 0},
        "test":  {"patients": 0, "slices": 0},
    }


def save_slices(volume_img, volume_lbl, split, pid, base_out):
    """Save 2D slices and return number of slices."""
    depth = volume_img.shape[0]

    for i in range(depth):
        name = f"{pid}_{i+1:03d}.png"

        Image.fromarray(volume_img[i]).save(
            os.path.join(base_out, "images", split, name)
        )
        Image.fromarray(volume_lbl[i]).save(
            os.path.join(base_out, "labels", split, name)
        )

    return depth


def print_stats(dataset_name, stats):
    """Pretty-print dataset statistics."""
    print(f"\n[{dataset_name} Summary]")
    total_p, total_s = 0, 0
    for split in ["train", "val", "test"]:
        p = stats[split]["patients"]
        s = stats[split]["slices"]
        total_p += p
        total_s += s
        print(f"  {split:5s}: {p:3d} patients | {s:6d} slices")

    print(f"  TOTAL : {total_p:3d} patients | {total_s:6d} slices\n")


def find_isles_dwi(dwi_root):
    """
    ISLES-2022 DWI structure:
    dwi/
      └── *_dwi.nii/        (directory!)
          └── *.nii         (actual image)
    """
    if not os.path.isdir(dwi_root):
        return None

    # Look for *_dwi.nii directories
    for d in os.listdir(dwi_root):
        if d.lower().endswith("_dwi.nii"):
            dwi_dir = os.path.join(dwi_root, d)
            if not os.path.isdir(dwi_dir):
                continue

            # Inside that directory, find the actual NIfTI
            for f in os.listdir(dwi_dir):
                f_low = f.lower()
                if f_low.endswith((".nii", ".nii.gz")) and "dwi" in f_low:
                    return os.path.join(dwi_dir, f)

    return None


# ============================================================
# ISLES 2022
# ============================================================
def make_isles22_dataset(isles_root, output_root):
    """
    ISLES 2022:
    - Input  : DWI (.nii)
    - Label  : Binary stroke mask (.nii)
    """

    random.seed(SEED)
    ensure_dirs(output_root)
    stats = init_stats()

    subjects = sorted([
        d for d in os.listdir(isles_root)
        if d.startswith("sub-strokecase")
    ])

    train_ids, val_ids, test_ids = split_patients(subjects)
    split_map = {pid: "train" for pid in train_ids}
    split_map.update({pid: "val" for pid in val_ids})
    split_map.update({pid: "test" for pid in test_ids})

    print(f"[ISLES22] Patients: {len(subjects)} | "
          f"Train {len(train_ids)} | Val {len(val_ids)} | Test {len(test_ids)}")

    for patient_idx, pid in enumerate(tqdm(subjects, desc="ISLES22"), start=1):
        split = split_map[pid]

        dwi_root = os.path.join(isles_root, pid, "ses-0001", "dwi")
        dwi_path = find_isles_dwi(dwi_root)

        if dwi_path is None:
            print(f"[ISLES22] Skipping {pid}: no valid DWI image found.")
            continue

        try:
            dwi = nib.load(dwi_path).get_fdata()
        except ImageFileError:
            print(f"[ISLES22] Skipping {pid}: invalid/corrupted DWI file.")
            continue

        mask_path = os.path.join(
            isles_root, "derivatives", pid, "ses-0001",
            f"{pid}_ses-0001_msk.nii"
        )

        dwi = nib.load(dwi_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        dwi = resize(dwi, TARGET_SHAPE, order=1, preserve_range=True, anti_aliasing=True)
        mask = resize(mask, TARGET_SHAPE, order=0, preserve_range=True, anti_aliasing=False)

        dwi = normalize_to_uint8(dwi)
        mask = (mask > 0).astype(np.uint8) * 255

        num_slices = save_slices(dwi, mask, split, patient_idx, output_root)
        stats[split]["patients"] += 1
        stats[split]["slices"] += num_slices

    print_stats("ISLES22", stats)


# ============================================================
# BraTS 2019
# ============================================================
def make_brats19_dataset(brats_root, output_root):
    """
    BraTS 2019:
    - Input  : FLAIR only
    - Label  : Binary tumor mask (all tumor classes merged)
    """

    random.seed(SEED)
    ensure_dirs(output_root)
    stats = init_stats()

    patients = []
    for grade in ["HGG", "LGG"]:
        grade_dir = os.path.join(brats_root, grade)
        for p in os.listdir(grade_dir):
            patients.append(os.path.join(grade, p))

    train_ids, val_ids, test_ids = split_patients(patients)
    split_map = {pid: "train" for pid in train_ids}
    split_map.update({pid: "val" for pid in val_ids})
    split_map.update({pid: "test" for pid in test_ids})

    print(f"[BraTS19] Patients: {len(patients)} | "
          f"Train {len(train_ids)} | Val {len(val_ids)} | Test {len(test_ids)}")

    for patient_idx, pid in enumerate(tqdm(patients, desc="BraTS19"), start=1):
        split = split_map[pid]

        base = os.path.join(brats_root, pid)
        name = os.path.basename(pid)

        flair_path = os.path.join(base, f"{name}_flair.nii")
        seg_path   = os.path.join(base, f"{name}_seg.nii")

        flair = nib.load(flair_path).get_fdata()
        seg   = nib.load(seg_path).get_fdata()

        flair = resize(flair, TARGET_SHAPE, order=1, preserve_range=True, anti_aliasing=True)
        seg   = resize(seg, TARGET_SHAPE, order=0, preserve_range=True, anti_aliasing=False)

        flair = normalize_to_uint8(flair)
        seg = (seg > 0).astype(np.uint8) * 255

        num_slices = save_slices(flair, seg, split, patient_idx, output_root)
        stats[split]["patients"] += 1
        stats[split]["slices"] += num_slices

    print_stats("BraTS19", stats)


if __name__ == "__main__":

    make_isles22_dataset(
        isles_root="data/isles22/3d_data/ISLES-2022",
        output_root="data/isles22/2d_data"
    )

    make_brats19_dataset(
        brats_root="data/brats19/3d_data/MICCAI_BraTS_2019_Data_Training",
        output_root="data/brats19/2d_data"
    )
