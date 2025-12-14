import os
import csv
import time
import shutil
import logging
import datetime
from typing import Dict, Tuple
from collections import defaultdict
import numpy as np
import imageio
import torch
import torch.optim as optim
from tqdm import tqdm

from SSL4MIS.networks.unet import UNet
from utils.dataload import BrainSegmentationDatasetBBox, load_bbox_data, split_labels_TwoStream
from utils.loss import combined_gan_loss, dice_loss, discriminator_loss, generator_loss
from utils.helpers import (
    save_training_plots_from_csv_G1,
    compute_gradient_penalty,
    update_ema_variables,
    log_epoch_metrics,
    add_gradient_noise,
    compute_model_complexity,
    load_yaml_config,
    set_seed,
    setup_logger,
    ensure_one_mode,
    get_device,
)
from utils.discriminator import FC5_Discriminator
from utils.metrics import test_single_case, calculate_metric_percase

def get_input_size(with_bbox: bool, h: int, w: int) -> Tuple[int, int, int, int]:
    # your training uses DWI + (BBox) + noise => 3 channels, or DWI + noise => 2 channels
    c = 3 if with_bbox else 2
    return (1, c, h, w)

def build_models(with_bbox: bool, device: torch.device):
    D0 = FC5_Discriminator().to(device)
    D1 = FC5_Discriminator().to(device)

    in_ch = 3 if with_bbox else 2
    G0 = UNet(in_chns=in_ch, class_num=1).to(device)
    G1 = UNet(in_chns=in_ch, class_num=1).to(device)
    G2 = UNet(in_chns=in_ch, class_num=1).to(device)

    return G0, G1, G2, D0, D1

def build_optimizers(G0, G1, G2, D0, D1, lr: float):
    opt_G0 = optim.Adam(G0.parameters(), lr=lr)
    opt_G1 = optim.Adam(G1.parameters(), lr=lr)
    opt_G2 = optim.Adam(G2.parameters(), lr=lr)
    opt_D0 = optim.Adam(D0.parameters(), lr=lr)
    opt_D1 = optim.Adam(D1.parameters(), lr=lr)
    return opt_G0, opt_G1, opt_G2, opt_D0, opt_D1

def write_run_header(
    logger: logging.Logger,
    eval_log_path: str,
    cfg: dict,
    device: torch.device,
    model_name: str,
    params: str,
    flops: str,
) -> None:
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    sys_cfg = cfg["system"]

    with open(eval_log_path, "a") as f:
        f.write(f"Start Time: {t}\n")
        f.write(f"Experiment: {cfg['experiment']['name']}\n")
        f.write(f"Batch Size: {train_cfg['batch_size']}\n")
        f.write(f"Labeled Ratio: {train_cfg['labeled_ratio'] * 100:.2f}%\n")
        f.write(f"Epochs: {train_cfg['epochs']}\n")
        f.write(f"LR: {train_cfg['lr']}\n")
        f.write(f"Device: {device}\n")
        f.write(f"AMP: {sys_cfg.get('amp', False)}\n")
        f.write(f"Input Mode: {'DWI+BBox+Noise' if model_cfg['with_bbox'] else 'DWI+Noise'}\n")
        f.write(f"Generator: {model_name}\n")
        f.write(f"Discriminator: FC5_Discriminator\n")
        f.write(f"Params = {params}, FLOPs = {flops}\n\n")

    logger.info("Run initialized. Logging header written.")


def init_metrics_csv(metrics_file_path: str) -> None:
    os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)
    with open(metrics_file_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "G1_train_loss", "G1_val_loss", "G1_3d_dsc"])


# ============================================================
# DATA
# ============================================================
def build_datasets_and_loaders(cfg: dict):
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    dwi_train = os.path.join(data_cfg["dwi_dir"], data_cfg["split_train"])
    mask_train = os.path.join(data_cfg["mask_dir"], data_cfg["split_train"])
    dwi_val = os.path.join(data_cfg["dwi_dir"], data_cfg["split_val"])
    mask_val = os.path.join(data_cfg["mask_dir"], data_cfg["split_val"])

    train_dataset = BrainSegmentationDatasetBBox(
        dwi_train,
        mask_train,
        load_bbox_data(data_cfg["bbox_train_json"]),
    )

    val_dataset = BrainSegmentationDatasetBBox(
        dwi_val,
        mask_val,
        load_bbox_data(data_cfg["bbox_val_json"]),
    )

    batch_size = int(train_cfg["batch_size"])
    labeled_batch_size = int(batch_size * float(train_cfg.get("labeled_batch_fraction", 0.5)))

    patientsID_p = os.path.join(cfg["experiment"]["output_dir"], "patientsID.txt")
    os.makedirs(cfg["experiment"]["output_dir"], exist_ok=True)

    train_loader, val_loader = split_labels_TwoStream(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        label_size=float(train_cfg["labeled_ratio"]),
        patID=patientsID_p,
        labeled_batch_size=labeled_batch_size,
    )

    return train_dataset, val_dataset, train_loader, val_loader, labeled_batch_size


# ============================================================
# TRAINING
# ============================================================
def make_inputs(images: torch.Tensor, with_bbox: bool, noise_std: float) -> torch.Tensor:
    """
    images: [B, C, H, W] where C is at least 2 if with_bbox else at least 1.
    """
    dwi = images[:, 0:1, :, :]  # always DWI in channel 0
    noise = torch.randn_like(dwi) * noise_std

    if with_bbox:
        bbox = images[:, 1:2, :, :]
        return torch.cat([dwi, bbox, noise], dim=1)  # [B, 3, H, W]

    return torch.cat([dwi, noise], dim=1)  # [B, 2, H, W]


def train_one_epoch(
    epoch: int,
    cfg: dict,
    device: torch.device,
    train_loader,
    labeled_batch_size: int,
    G0, G1, G2, D0, D1,
    opt_G0, opt_G1, opt_G2, opt_D0, opt_D1,
    global_step: int,
    logger: logging.Logger,
) -> Tuple[float, int]:
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    G0.train(); D0.train(); G1.train(); D1.train(); G2.train()

    consistency_rampup_epoch = int(train_cfg["consistency_rampup_epoch"])
    lambda_gp = float(train_cfg["lambda_gp"])
    noise_std = float(train_cfg["noise_std"])
    ema_alpha = float(train_cfg["ema_alpha"])
    with_bbox = bool(model_cfg["with_bbox"])

    epoch_d0_loss = 0.0
    epoch_g0_loss = 0.0
    epoch_d1_loss = 0.0
    epoch_g1_loss = 0.0

    consistency_weight = min(1.0, epoch / max(1, consistency_rampup_epoch))

    for images, masks in tqdm(train_loader, desc=f"Train {epoch+1}/{train_cfg['epochs']}"):
        images = images.to(device)
        masks = masks.to(device)

        images_labeled = images[:labeled_batch_size]
        masks_labeled = masks[:labeled_batch_size]
        images_unlabeled = images[labeled_batch_size:]

        inputs_labeled = make_inputs(images_labeled, with_bbox=with_bbox, noise_std=noise_std)
        inputs_unlabeled = make_inputs(images_unlabeled, with_bbox=with_bbox, noise_std=noise_std)

        # === Pseudo-labels ===
        with torch.no_grad():
            pseudo_gt_g0 = torch.sigmoid(G0(inputs_unlabeled).detach())
            pseudo_gt_g2 = torch.sigmoid(G2(inputs_unlabeled).detach())

        # === Train D0 ===
        opt_D0.zero_grad()
        real_scores_d0 = D0(masks_labeled)
        fake_g0 = G0(inputs_labeled).detach()
        fake_scores_d0 = D0(fake_g0)
        gp_d0 = compute_gradient_penalty(D0, masks_labeled, fake_g0)
        d0_loss = discriminator_loss(real_scores_d0, fake_scores_d0) + lambda_gp * gp_d0
        d0_loss.backward()
        opt_D0.step()

        # === Train G0 ===
        opt_G0.zero_grad()
        g0_output = G0(inputs_labeled)
        adv_loss_g0 = generator_loss(D0(g0_output))
        g0_loss = combined_gan_loss(g0_output, masks_labeled) + adv_loss_g0
        g0_loss.backward()
        add_gradient_noise(G0, stddev=1e-5)
        opt_G0.step()

        # === Train D1 (unlabeled) ===
        opt_D1.zero_grad()
        real_scores_d1 = D1(pseudo_gt_g2)
        g1_fake = G1(inputs_unlabeled).detach()
        fake_scores_d1 = D1(g1_fake)
        gp_d1 = compute_gradient_penalty(D1, pseudo_gt_g2, g1_fake)
        d1_supervised_loss = discriminator_loss(real_scores_d1, fake_scores_d1) + lambda_gp * gp_d1

        with torch.no_grad():
            d0_pseudo = D0(pseudo_gt_g2)
            d0_fake_g1 = D0(g1_fake)

        loss_consistency_d1 = (
            dice_loss(torch.sigmoid(real_scores_d1), torch.sigmoid(d0_pseudo.detach())) +
            dice_loss(torch.sigmoid(fake_scores_d1), torch.sigmoid(d0_fake_g1.detach()))
        )
        d1_loss = d1_supervised_loss + consistency_weight * loss_consistency_d1
        d1_loss.backward()
        opt_D1.step()

        # === Train G1 (unlabeled) ===
        opt_G1.zero_grad()
        g1_output = G1(inputs_unlabeled)

        loss_consistency_g1 = dice_loss(torch.sigmoid(g1_output), pseudo_gt_g0)
        adv_loss_g1 = generator_loss(D1(g1_output))
        g1_loss = combined_gan_loss(g1_output, pseudo_gt_g0) + adv_loss_g1 + consistency_weight * loss_consistency_g1

        # EMA consistency with G2
        loss_consistency_ema = dice_loss(torch.sigmoid(g1_output), pseudo_gt_g2)
        g1_loss = g1_loss + consistency_weight * loss_consistency_ema

        g1_loss.backward()
        add_gradient_noise(G1, stddev=1e-5)
        opt_G1.step()

        update_ema_variables(G1, G2, alpha=ema_alpha, global_step=global_step)

        global_step += 1
        epoch_d0_loss += d0_loss.item()
        epoch_g0_loss += g0_loss.item()
        epoch_d1_loss += d1_loss.item()
        epoch_g1_loss += g1_loss.item()

    avg_g1 = epoch_g1_loss / max(1, len(train_loader))
    logger.info(
        f"[Epoch {epoch+1}] Train losses | D0: {epoch_d0_loss/len(train_loader):.4f} "
        f"G0: {epoch_g0_loss/len(train_loader):.4f} D1: {epoch_d1_loss/len(train_loader):.4f} "
        f"G1: {avg_g1:.4f} | cw={consistency_weight:.3f}"
    )
    return avg_g1, global_step


# ============================================================
# EVALUATION
# ============================================================
def evaluate_3d_metrics(
    cfg: dict,
    device: torch.device,
    val_dataset,
    test_names,
    model,
    logger: logging.Logger,
) -> Tuple[float, float, float, float, Dict[str, np.ndarray], int, int, int]:
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    threshold = float(train_cfg["threshold"])
    noise_std = float(train_cfg["noise_std"])
    with_bbox = bool(model_cfg["with_bbox"])

    model.eval()

    patient_slices_pred = defaultdict(dict)
    patient_slices_gt = defaultdict(dict)
    total_slices = 0
    epoch_pred_masks: Dict[str, np.ndarray] = {}

    with torch.no_grad():
        for idx, (image, mask) in enumerate(val_dataset):
            name = test_names[idx]
            image = image.to(device)
            mask = mask.to(device)

            if image.dim() == 3:
                image = image.unsqueeze(0)
            if mask.dim() == 3:
                mask = mask.unsqueeze(0)

            dwi = image[:, 0:1, :, :]
            noise = torch.randn_like(dwi) * noise_std

            if with_bbox:
                bbox = image[:, 1:2, :, :]
                model_input = torch.cat([dwi, bbox, noise], dim=1)
            else:
                model_input = torch.cat([dwi, noise], dim=1)

            output = model(model_input)
            probs = torch.sigmoid(output)
            pred_bin = (probs > threshold).float()
            gt_bin = (mask > 0.5).float()

            total_slices += 1

            pred_np = pred_bin.cpu().numpy().squeeze().astype(np.uint8)
            gt_np = gt_bin.cpu().numpy().squeeze().astype(np.uint8)

            epoch_pred_masks[name] = pred_np

            patient_id, slice_idx = name.split("_")
            patient_id, slice_idx = int(patient_id), int(slice_idx)
            patient_slices_pred[patient_id][slice_idx] = pred_np
            patient_slices_gt[patient_id][slice_idx] = gt_np

    # Per-patient 3D metrics using your existing utilities
    total_dice, total_jc, total_hd, total_asd = [], [], [], []
    for pid in sorted(patient_slices_pred):
        pred_slices = patient_slices_pred[pid]
        gt_slices = patient_slices_gt[pid]
        max_slice = max(pred_slices.keys())

        ref_pred = next(iter(pred_slices.values()))
        ref_zero = np.zeros_like(ref_pred, dtype=np.uint8)

        pred_stack = np.stack([pred_slices.get(i, ref_zero) for i in range(max_slice + 1)], axis=0)
        gt_stack = np.stack([gt_slices.get(i, ref_zero) for i in range(max_slice + 1)], axis=0)

        score_map = np.zeros((2,) + pred_stack.shape, dtype=np.float32)
        score_map[1] = pred_stack.astype(np.float32)
        score_map[0] = 1.0 - score_map[1]

        pred_3d = test_single_case(score_map)
        gt_3d = gt_stack.astype(np.uint8)

        dice, jc, hd, asd = calculate_metric_percase(pred_3d, gt_3d)
        total_dice.append(dice)
        total_jc.append(jc)
        total_hd.append(hd)
        total_asd.append(asd)

    mean_3d_dsc = float(np.mean(total_dice)) if len(total_dice) else 0.0
    mean_3d_jc = float(np.mean(total_jc)) if len(total_jc) else 0.0
    mean_3d_hd = float(np.mean(total_hd)) if len(total_hd) else 0.0
    mean_3d_asd = float(np.mean(total_asd)) if len(total_asd) else 0.0

    logger.info(
        f"Eval summary | slices={total_slices} patients={len(patient_slices_pred)} "
        f"3D: DSC={mean_3d_dsc*100:.2f}% IoU={mean_3d_jc*100:.2f}% HD95={mean_3d_hd:.2f} ASD={mean_3d_asd:.2f}"
    )

    return mean_3d_dsc, mean_3d_jc, mean_3d_hd, mean_3d_asd, epoch_pred_masks, total_slices, len(patient_slices_pred)


def save_pred_masks_png(epoch_pred_masks: Dict[str, np.ndarray], out_dir: str) -> None:
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    for name, pred_np in epoch_pred_masks.items():
        imageio.imwrite(os.path.join(out_dir, f"{name}.png"), (pred_np.astype(np.uint8) * 255))


# ============================================================
# MAIN
# ============================================================
def main():
    cfg = load_yaml_config("config/config_train.yaml")

    ensure_one_mode(cfg["model"]["with_bbox"])

    out_dir = cfg["experiment"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    eval_log_path = os.path.join(out_dir, "val_metrics.txt")
    metrics_file_path = os.path.join(out_dir, "loss_metrics.csv")
    log_file = os.path.join(out_dir, "train.log")

    logger = setup_logger(log_file)

    set_seed(int(cfg["experiment"]["seed"]))

    device = get_device(cfg["system"]["device"])
    logger.info(f"Using device: {device}")

    # Data
    train_dataset, val_dataset, train_loader, val_loader, labeled_batch_size = build_datasets_and_loaders(cfg)

    # Build test_names consistent with your dataset
    val_split_dir = os.path.join(cfg["data"]["dwi_dir"], cfg["data"]["split_val"])
    test_names = sorted(os.listdir(val_split_dir))
    test_names = [os.path.splitext(f)[0] for f in test_names]

    # Models + Optimizers
    G0, G1, G2, D0, D1 = build_models(cfg["model"]["with_bbox"], device)
    opt_G0, opt_G1, opt_G2, opt_D0, opt_D1 = build_optimizers(G0, G1, G2, D0, D1, float(cfg["training"]["lr"]))

    # Complexity
    h, w = cfg["model"]["image_size"]
    input_size = get_input_size(cfg["model"]["with_bbox"], h, w)
    flops, params = compute_model_complexity(G1, input_size, device)

    write_run_header(
        logger=logger,
        eval_log_path=eval_log_path,
        cfg=cfg,
        device=device,
        model_name=G1.__class__.__name__,
        params=params,
        flops=flops,
    )

    init_metrics_csv(metrics_file_path)

    # Training loop
    num_epochs = int(cfg["training"]["epochs"])
    best_3d_dsc = -1.0
    global_step = 0

    start_time = time.time()
    for epoch in range(num_epochs):
        avg_g1_loss, global_step = train_one_epoch(
            epoch=epoch,
            cfg=cfg,
            device=device,
            train_loader=train_loader,
            labeled_batch_size=labeled_batch_size,
            G0=G0, G1=G1, G2=G2, D0=D0, D1=D1,
            opt_G0=opt_G0, opt_G1=opt_G1, opt_G2=opt_G2, opt_D0=opt_D0, opt_D1=opt_D1,
            global_step=global_step,
            logger=logger,
        )

        # Evaluate with EMA teacher (G2), matching your script
        mean_3d_dsc, mean_3d_jc, mean_3d_hd, mean_3d_asd, epoch_pred_masks, total_slices, total_patients = evaluate_3d_metrics(
            cfg=cfg,
            device=device,
            val_dataset=val_dataset,
            test_names=test_names,
            model=G2,
            logger=logger,
        )

        # Log to evaluation text file
        with open(eval_log_path, "a") as f:
            f.write(
                f"[Epoch {epoch+1}] Slices: {total_slices}, Patients: {total_patients} || "
                f"DSC: {mean_3d_dsc * 100:.2f}%, IoU: {mean_3d_jc * 100:.2f}%, "
                f"HD95: {mean_3d_hd:.2f}, ASD: {mean_3d_asd:.2f}\n"
            )

        # Save best
        if mean_3d_dsc > best_3d_dsc:
            best_3d_dsc = mean_3d_dsc
            logger.info(f"[New Best] 3D DSC = {best_3d_dsc*100:.2f}%")

            torch.save(G2.state_dict(), os.path.join(out_dir, "model_best.pth"))

            best_dir = os.path.join(out_dir, "segmentation_outputs", "best_epoch")
            save_pred_masks_png(epoch_pred_masks, best_dir)

            with open(eval_log_path, "a") as f:
                f.write(f"[New Best] DSC: {best_3d_dsc*100:.2f}%\n")

        # CSV plotting/logging (kept with your helper)
        log_epoch_metrics(
            metrics_file_path,
            epoch + 1,
            G1_train_loss=float(avg_g1_loss),
            G1_val_loss=0.0,
            G1_3d_dsc=float(mean_3d_dsc * 100.0),
        )

    end_time = time.time()
    hours = (end_time - start_time) / 3600.0
    logger.info(f"Training finished in {hours:.2f} hours.")

    save_training_plots_from_csv_G1(metrics_file_path, out_dir)

    with open(eval_log_path, "a") as f:
        f.write(f"\nTotal Training Duration: {hours:.2f} hours.\n")
        f.write(f"Experiment: {cfg['experiment']['name']}\n")


if __name__ == "__main__":
    main()
