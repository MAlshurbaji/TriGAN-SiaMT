import os
import json
import random
import cv2
from itertools import combinations
from tqdm import tqdm

"""
Bounding Box Generation Script
==============================
- Extracts lesion bounding boxes from binary mask images
- Supports ISLES22 and BraTS19
- Processes train / val / test splits
- Saves:
  1) mask images with bounding boxes overlaid
  2) bounding boxes in JSON format
"""

# ============================================================
# BBOX UTILITIES
# ============================================================
def merge_intersecting_bboxes(bboxes):
    """Merge overlapping bounding boxes into one."""
    merged = list(bboxes)

    while True:
        new_merged = []
        used = set()

        for i, j in combinations(range(len(merged)), 2):
            if i in used or j in used:
                continue

            x1, y1, w1, h1 = merged[i]
            x2, y2, w2, h2 = merged[j]

            if (x1 < x2 + w2 and x1 + w1 > x2 and
                y1 < y2 + h2 and y1 + h1 > y2):

                x_new = min(x1, x2)
                y_new = min(y1, y2)
                w_new = max(x1 + w1, x2 + w2) - x_new
                h_new = max(y1 + h1, y2 + h2) - y_new

                new_merged.append((x_new, y_new, w_new, h_new))
                used.update([i, j])

        for i, bbox in enumerate(merged):
            if i not in used:
                new_merged.append(bbox)

        if len(new_merged) == len(merged):
            break

        merged = new_merged

    return merged


def extract_bboxes_from_mask(mask_path, expansion_range=(0, 0)):
    """Extract lesion bounding boxes from a binary mask image."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read image: {mask_path}")

    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        expansion = random.randint(*expansion_range)

        x = max(0, x - expansion)
        y = max(0, y - expansion)
        w = min(mask.shape[1] - x, w + 2 * expansion)
        h = min(mask.shape[0] - y, h + 2 * expansion)

        bboxes.append((x, y, w, h))

    return merge_intersecting_bboxes(bboxes)


def save_mask_with_bboxes(mask_path, bboxes, output_dir):
    """Save the mask image with bounding boxes overlaid."""
    os.makedirs(output_dir, exist_ok=True)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    for x, y, w, h in bboxes:
        cv2.rectangle(mask, (x, y), (x + w - 1, y + h - 1), 255, 1)

    cv2.imwrite(os.path.join(output_dir, os.path.basename(mask_path)), mask)


def save_bboxes_json(bboxes_dict, json_path):
    """Save bounding boxes to a JSON file."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(bboxes_dict, f, indent=4)


# ============================================================
# DATASET PROCESSOR
# ============================================================
def process_dataset(dataset_name, label_root, bbox_root, expansion_range):
    """
    Process train/val/test splits for a dataset.
    """
    print(f"\n[{dataset_name}] Generating bounding boxes")

    for split in ["train", "val", "test"]:
        print(f"  → Split: {split}")

        mask_dir = os.path.join(label_root, split)
        bbox_mask_dir = os.path.join(bbox_root, "mask_bbox", split)
        json_path = os.path.join(bbox_root, f"bbox_{split}.json")

        all_bboxes = {}

        for fname in tqdm(sorted(os.listdir(mask_dir)), desc=f"{dataset_name}-{split}"):
            if not fname.lower().endswith(".png"):
                continue

            mask_path = os.path.join(mask_dir, fname)

            try:
                bboxes = extract_bboxes_from_mask(mask_path, expansion_range)
            except FileNotFoundError as e:
                print(e)
                continue

            all_bboxes[fname] = bboxes
            save_mask_with_bboxes(mask_path, bboxes, bbox_mask_dir)

        save_bboxes_json(all_bboxes, json_path)

        print(f"    Saved: {len(all_bboxes)} masks")
        print(f"    JSON : {json_path}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    EXPANSION_RANGE = (1, 1)  # fixed 1 pixel expansion, can be adjusted to (1,3) for random between 1 to 3 pixels, etc..

    # ----------------------------
    # ISLES22
    # ----------------------------
    process_dataset(
        dataset_name="ISLES22",
        label_root="data/isles22/2d_data/labels",
        bbox_root="data/isles22/bboxes",
        expansion_range=EXPANSION_RANGE,
    )

    # ----------------------------
    # BraTS19
    # ----------------------------
    process_dataset(
        dataset_name="BraTS19",
        label_root="data/brats19/2d_data/labels",
        bbox_root="data/brats19/bboxes",
        expansion_range=EXPANSION_RANGE,
    )
