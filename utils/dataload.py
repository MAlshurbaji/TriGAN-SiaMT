'''
ISLES 2022 sets:
	Train      -- 187
	Validation -- 25
	AllData    -- 212
	Test       -- 38
'''
import os, math, torch, cv2, json, random, sys
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class BrainSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.image_dir = image_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.image_dir, self.image_filenames[idx]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(self.mask_dir, self.mask_filenames[idx]), cv2.IMREAD_GRAYSCALE)
        img, mask = img.astype(np.float32) / 255.0, mask.astype(np.float32) / 255.0
        return torch.tensor(img).unsqueeze(0), torch.tensor(mask).unsqueeze(0)

def load_bbox_data(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def create_bbox_heatmap(image_size, bbox_list):
    """
    Create a heatmap for bounding boxes.
    - image_size: Tuple (H, W)
    - bbox_list: List of bounding boxes [[x, y, w, h], ...]
    """
    heatmap = np.zeros(image_size, dtype=np.float32)
    for bbox in bbox_list:
        x, y, w, h = bbox
        heatmap[y:y+h, x:x+w] = 1.0  # Fill bbox region with 1s

    return torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)  # Shape (1, H, W)

# DataLoader with Bounding Box Processing
class BrainSegmentationDatasetBBox(BrainSegmentationDataset):
    def __init__(self, image_dir, mask_dir, bbox_data, pre_trained=True):
        super().__init__(image_dir, mask_dir)
        self.bbox_data = bbox_data  # Dictionary with bbox annotations
        self.pre_trained = pre_trained

    def __getitem__(self, idx):
        image, mask = super().__getitem__(idx)  # Get standard image & mask
        img_name = self.image_filenames[idx]
        # Retrieve bbox list (default empty if no bboxes)
        bbox_list = self.bbox_data.get(img_name, [])
        bbox_heatmap = create_bbox_heatmap(image.shape[1:], bbox_list) # Create BBox Heatmap
        if self.pre_trained:
            return torch.cat((image, bbox_heatmap), dim=0), mask  # Add bbox as additional channel
        else:
            return {'image': torch.cat((image, bbox_heatmap), dim=0), 'label': mask, 'filename': img_name}

def split_patient_indices(dataset, labeled_ratio, seed=42, output_txt_path=None):
    """
    Splits dataset into labeled and unlabeled by patient ID.
    Optionally logs filenames for both sets to a file.
    """
    random.seed(seed)
    patient_to_slices = defaultdict(list)
    # Group slice filenames by patient ID
    for idx, filename in enumerate(dataset.image_filenames):
        if "_" in filename:
            patient_id = filename.split("_")[0]
            patient_to_slices[patient_id].append((idx, filename))
    all_patients = sorted(patient_to_slices.keys())
    random.shuffle(all_patients)

    num_labeled_patients = math.ceil(labeled_ratio * len(all_patients))
    labeled_patients = set(all_patients[:num_labeled_patients])
    labeled_indices, unlabeled_indices = [], []
    labeled_files, unlabeled_files = [], []

    for patient_id, slice_data in patient_to_slices.items():
        if patient_id in labeled_patients:
            for idx, fname in slice_data:
                labeled_indices.append(idx)
                labeled_files.append(fname)
        else:
            for idx, fname in slice_data:
                unlabeled_indices.append(idx)
                unlabeled_files.append(fname)

    # === Optional logging ===
    if output_txt_path:
        with open(output_txt_path, "w") as f:
            f.write("=== Labeled Files ===\n")
            for fname in labeled_files:
                f.write(f"{fname}\n")
            f.write("\n=== Unlabeled Files ===\n")
            for fname in unlabeled_files:
                f.write(f"{fname}\n")

    return labeled_indices, unlabeled_indices

from SSL4MIS.dataloaders import TwoStreamBatchSampler
def split_labels_TwoStream(train_dataset, val_dataset, batch_size, label_size, labeled_batch_size, num_workers=8, patID=None):
    # === Split labeled/unlabeled indices based on patient ID ===
    labeled_indices, unlabeled_indices = split_patient_indices(dataset=train_dataset, labeled_ratio=label_size, output_txt_path=patID)
    batch_sampler = TwoStreamBatchSampler(primary_indices=labeled_indices, secondary_indices=unlabeled_indices, 
                                          batch_size=batch_size, secondary_batch_size=batch_size-labeled_batch_size)

    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
