import torch
import pandas as pd
import matplotlib.pyplot as plt
import os, csv
import random
import numpy as np
import logging
from thop import profile, clever_format

def save_training_plots_from_csv(metrics_csv_path, output_dir):
    df = pd.read_csv(metrics_csv_path)

    # Plot G2
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df['epoch'], df['G2_val_loss (1-avg_dsc_all)'], 'b-', label='G2 Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('G2 Validation Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(df['epoch'], df['G2_3d_dsc'], color='r', label='G2 3D-DSC')
    ax2.set_ylabel('G2 3D-DSC (%)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.suptitle('G2 Validation Loss and 3D-DSC per Epoch', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'G2_val_loss_and_3d_dsc.png'), dpi=300)
    plt.close()

    # Plot G1
    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df['G1_train_loss'], label='G1 Train Loss', color='navy')
    plt.plot(df['epoch'], df['G1_val_loss'], label='G1 Val Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('G1 Training and Validation Loss per Epoch', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'G1_train_val_loss.png'), dpi=300)
    plt.close()

def save_training_plots_from_csv_G1(metrics_csv_path, output_dir):
    df = pd.read_csv(metrics_csv_path)

    # Single plot with G1_train_loss, G1_val_loss, G1_3d_dsc
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(df['epoch'], df['G1_train_loss'], label='G1 Train Loss', color='navy')
    ax1.plot(df['epoch'], df['G1_val_loss'], label='G1 Val Loss', color='orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    ax2 = ax1.twinx()
    ax2.plot(df['epoch'], df['G1_3d_dsc'], 'r--', label='G1 3D-DSC')
    ax2.set_ylabel('G1 3D-DSC (%)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.suptitle('G1 Training Loss, Validation Loss, and 3D-DSC per Epoch', fontsize=14)

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'G1_train_val_loss_and_3d_dsc.png'), dpi=300)
    plt.close()

def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones_like(d_interpolates)
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=fake, 
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)  # Ramp-up
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def log_epoch_metrics(metrics_file_path, epoch, G1_train_loss, G1_val_loss, G1_3d_dsc):
    with open(metrics_file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, G1_train_loss, G1_val_loss, G1_3d_dsc])

def add_gradient_noise(model, stddev=1e-5):
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * stddev
            param.grad += noise

def compute_model_complexity(model, input_size, device):
    """
    Compute FLOPs and number of parameters for a PyTorch model using THOP.
    """
    dummy_input = torch.randn(*input_size).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops, params = clever_format([flops, params], "%.2f")
    return flops, params



def load_yaml_config(path: str) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("Missing dependency: pyyaml. Install with `pip install pyyaml`.") from e

    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic may slow down; keep benchmark True for speed unless strict reproducibility is needed
    torch.backends.cudnn.benchmark = True

def setup_logger(log_file: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger

def ensure_one_mode(with_bbox: bool) -> None:
    if not isinstance(with_bbox, bool):
        raise ValueError("model.with_bbox must be a boolean.")
    # Single switch: either bbox-mode or no-bbox-mode (mutually exclusive by design)

def get_device(device_str: str) -> torch.device:
    if torch.cuda.is_available() and device_str.startswith("cuda"):
        return torch.device(device_str)
    return torch.device("cpu")
