import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerConfig, get_polynomial_decay_schedule_with_warmup
from torch.utils.data import DataLoader
import evaluate
import os
import shutil
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import time
from metrics import Metrics  # Assuming metrics.py exists and contains the necessary classes
import pandas as pd
import argparse
import math
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib
import json
from my_datasets import CityscapesDataset
import matplotlib.pyplot as plt
import datasets
# ----------------------------------------------------------------------------------------------------------------------
class OHEMCrossEntropy2D(nn.Module):
    # This class is unused in the provided code but kept for completeness
    def __init__(self, topk=0.7, ignore_index=255, n_min=100000):
        super(OHEMCrossEntropy2D, self).__init__()
        self.topk = topk
        self.ignore_index = ignore_index
        self.n_min = n_min

    def forward(self, input, target):
        if target.dim() == 4:
            target = target.squeeze(1)
        pixel_losses = F.cross_entropy(
            input,
            target,
            reduction='none',
            ignore_index=self.ignore_index
        )
        pixel_losses_flatten = pixel_losses.view(-1)
        topk_num = max(int(self.topk * pixel_losses_flatten.numel()), self.n_min)
        top_k_losses, _ = torch.topk(pixel_losses_flatten, topk_num)
        loss = top_k_losses.mean()
        return loss


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, thresh: float = 0.7,
                 aux_weights: list = [1, 1]) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        n_min = labels[labels != self.ignore_label].numel() // preds.shape[1]
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return torch.mean(loss_hard)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


# def CF_loss(loss_basic, preds, labels, CM_Weights):
def OHEM_loss(loss_basic, preds, labels, CM_Weights):
    # preds in shape [B, C, H, W]
    n_classes = CM_Weights.shape[0]
    ignore_label = 255

    preds_per = preds.permute(0, 2, 3, 1)  # [n_batch, height, width, n_classes]
    predictions = preds_per.reshape(-1, n_classes)  # [n_pixels, n_classes]
    lbl = torch.flatten(labels)  # [n_pixels,]

    # Remove ignore labels
    mask = lbl != ignore_label
    predictions = predictions[mask]
    lbl = lbl[mask]

    loss = loss_basic(predictions, lbl)

    return loss


# def CF_loss(loss_basic, preds, labels, CM_Weights):
def CAL_loss(loss_basic, preds, labels, CM_Weights):
    # preds in shape [B, C, H, W]
    n_classes = CM_Weights.shape[0]
    ignore_label = 255
    eps = torch.tensor(1e-6).to(preds.device)

    preds_per = preds.permute(0, 2, 3, 1)  # [n_batch, height, width, n_classes]
    predictions = preds_per.reshape(-1, n_classes)  # [n_pixels, n_classes]
    lbl = torch.flatten(labels)  # [n_pixels,]

    # Remove ignore labels
    # Remove ignore labels
    mask = lbl != ignore_label
    predictions = predictions[mask]
    lbl = lbl[mask]
    loss1 = loss_basic(predictions, lbl)

    if CM_Weights.shape[0] != n_classes:
        raise ValueError(
            f"CM_Weights must be {n_classes}x{n_classes} for Cityscapes (19 classes), but got {CM_Weights.shape[0]}x{CM_Weights.shape[1]}")

    CM_Weight_tensor = torch.from_numpy(CM_Weights).to(preds.device).float()

    # CAL loss: each pixel prediction is weighted by the lbl row of the weight matrix
    mask_eye = torch.eye(n_classes).to(preds.device)  # Eye matrix of [n_classes, n_classes]
    mask_one_hot = mask_eye[lbl]  # one hot encoding of size [n_predictions, n_classes]
    mask_zero_hot = 1 - mask_one_hot  # zero hot encoding

    # Loss2 calculation logic
    loss_cal = -torch.sum(
        torch.log((1 - predictions) + eps) * mask_zero_hot * CM_Weight_tensor[lbl, :]
        , dim=1)

    #loss2 = -torch.sum(
    #    torch.log((1 - predictions + eps) * mask_zero_hot + (predictions + eps) * mask_one_hot) * CM_Weight_tensor[lbl, :],
    #    dim=1)

    # Apply the OHEM also to the CAL loss
    n_min = lbl.numel() // n_classes
    # loss_hard, _ = loss_cal.topk(n_min)
    loss2 = torch.mean(loss_cal)

    return loss1 + loss2
    # return loss2
# ----------------------------------------------------------------------------------------------------------------------
class OHEMCrossEntropy(nn.Module):
    """
    Standard OHEM for Semantic Segmentation.
    """

    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000):
        super().__init__()
        self.ignore_index = ignore_index
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.min_kept = min_kept
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits, labels):
        # Softmax if input is raw logits
        loss = self.criterion(logits, labels).view(-1)

        # Binary mask for hard examples
        loss_hard = loss[loss > self.thresh.to(loss.device)]

        if loss_hard.numel() < self.min_kept:
            # If not enough hard examples, take top-k
            loss_hard, _ = loss.topk(self.min_kept)

        return loss_hard.mean()


class CALoss(nn.Module):
    """
    Confusion-Aware Loss (CAL) implementation with OO design.
    """

    def __init__(self, n_classes=19, ignore_index=255, eps=1e-6):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.eps = eps
        # We don't use nn.CrossEntropy because CAL logic requires manual log-prob handling

    def forward(self, preds, labels, cm_weights):
        """
        preds: Tensor [B, C, H, W] - Assumed to be Probabilities (after Softmax)
        labels: Tensor [B, H, W]
        cm_weights: Numpy array or Tensor [C, C]
        """
        device = preds.device
        B, C, H, W = preds.shape

        # Reshape and Mask
        preds = preds.permute(0, 2, 3, 1).reshape(-1, C)
        labels = labels.view(-1)

        mask = labels != self.ignore_index
        preds = preds[mask]
        labels = labels[mask]

        if labels.numel() == 0:
            return torch.tensor(0.0, requires_grad=True).to(device)

        # Ensure CM Weights is a Tensor
        if not isinstance(cm_weights, torch.Tensor):
            cm_weights = torch.from_numpy(cm_weights).float().to(device)

        # Standard Cross Entropy part (Loss 1)
        # Gather the probabilities of the correct class
        target_probs = preds.gather(1, labels.unsqueeze(1)).squeeze()
        loss_ce = -torch.log(target_probs + self.eps).mean()

        # Confusion-Aware part (Loss 2)
        # mask_zero_hot: 1 for all classes EXCEPT the ground truth
        mask_eye = torch.eye(C, device=device)
        mask_zero_hot = 1 - mask_eye[labels]

        # Weighted log-likelihood of "not being the wrong class"
        # We use (1 - preds) because we want to penalize high probability on confusing classes
        confusion_penalty = -torch.sum(
            torch.log((1 - preds) + self.eps) * mask_zero_hot * cm_weights[labels, :],
            dim=1
        )

        return loss_ce + confusion_penalty.mean()
# ----------------------------------------------------------------------------------------------------------------------
def evaluate_model(model, dataloader, processor, metric, device, previous_cm):
    N_CLASSES_CITYSCAPES = 19
    metrics = Metrics(N_CLASSES_CITYSCAPES, 255, device)
    model.eval()
    total_eval_loss = 0.0
    num_batches = 0
    eval_loss_fn = OhemCrossEntropy()
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)  # Add leave=False to remove bar on completio

    with torch.no_grad():

        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # Resize to the model's width, but keep proportional height
            img_resized = F.interpolate(pixel_values, size=(512, 1024), mode='bilinear', align_corners=False)

            # Pad height from 512 to 1024 (256 pixels top and bottom)
            img_padded = F.pad(img_resized, (0, 0, 256, 256), value=0)

            outputs = model(pixel_values=img_padded)
            logits = outputs.logits

            # 3. Undo the Padding on Logits
            # Logits are 1/4 the size, so padding was 256/4 = 64
            logits_unpadded = logits[:, :, 64:192, :]  # Remove top/bottom 64 pixels

            upsampled_logits = F.interpolate(
                logits_unpadded,
                # logits,
                size=labels.shape[-2:],
                mode="bicubic",
                align_corners=False
            ).softmax(dim=1)

            eval_loss = CAL_loss(eval_loss_fn, upsampled_logits, labels, previous_cm)
            total_eval_loss += eval_loss.item()

            metrics.update(upsampled_logits, labels)

            predicted_masks = upsampled_logits.argmax(dim=1)
            metric.add_batch(
                predictions=predicted_masks.detach().cpu().numpy(),
                references=labels.detach().cpu().numpy()
            )
            num_batches += 1

    avg_eval_loss = total_eval_loss / num_batches if num_batches > 0 else 0.0
    # The compute_iou method returns (ious, miou, CM, ...)
    ious, miou, CM = metrics.compute_iou()  # CM is the Confusion Matrix
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()

    return miou, ious, CM, macc, acc, mf1, f1, avg_eval_loss

# ----------------------------------------------------------------------------------------------------------------------
def evaluate_cityscapes(model_name, dataloader, device="cuda"):
    # 1. Load Model, Processor, and Metric
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device)
    model.eval()

    metric = evaluate.load("mean_iou")

    # Cityscapes specific: ignore index 255 (void/background)
    # Most Cityscapes datasets use 19 classes for evaluation
    num_classes = 19

    print(f"Starting evaluation for {model_name}...")

    with torch.no_grad():
        for batch in tqdm(dataloader):
            # images: [Batch, 3, 1024, 1024], labels: [Batch, 1024, 1024]
            pixel_values = batch["pixel_values"].to(device)

            # 2. Scaling to 1024x1024
            # If your 512x512 is a stretched version of the 2:1 original,
            # the model needs to see that 2:1 ratio.
            # We resize to 1024x512 and pad to 1024x1024.

            # Resize to the model's width, but keep proportional height
            img_resized = F.interpolate(pixel_values, size=(512, 1024), mode='bilinear')

            # Pad height from 512 to 1024 (256 pixels top and bottom)
            img_padded = F.pad(img_resized, (0, 0, 256, 256), value=0)


            labels = batch["labels"].to(device)
            # pixel_values_1024 = F.interpolate(pixel_values, size=(1024, 1024), mode='bilinear', align_corners=False)

            # 2. Forward Pass
            outputs = model(pixel_values=img_padded)
            logits = outputs.logits  # Shape: [Batch, 19, 256, 256]

            # 3. Undo the Padding on Logits
            # Logits are 1/4 the size, so padding was 256/4 = 64
            logits_unpadded = logits[:, :, 64:192, :]  # Remove top/bottom 64 pixels

            # 3. Upsample Logits to Original Resolution (1024x1024)
            # This is critical. SegFormer outputs are 1/4 the input resolution.
            upsampled_logits = nn.functional.interpolate(
                logits_unpadded,
                size=labels.shape[-2:],  # Match the H, W of ground truth
                mode="bilinear",
                align_corners=False,
            )

            # 4. Get Predictions
            predictions = upsampled_logits.argmax(dim=1)

            # 5. Add to Metric
            # Note: evaluate expects numpy arrays
            metric.add_batch(
                predictions=predictions.detach().cpu().numpy(),
                references=labels.detach().cpu().numpy(),
            )

    # 6. Final Computation
    # ignore_index=255 ensures unlabeled pixels don't hurt your score
    results = metric.compute(
        num_labels=num_classes,
        ignore_index=255,
        reduce_labels=False
    )

    return results
# ----------------------------------------------------------------------------------------------------------------------
'''
def evaluate_model_sw(model, dataloader, processor, metric, device, previous_cm, loss_func):
    N_CLASSES_CITYSCAPES = 19
    metrics = Metrics(N_CLASSES_CITYSCAPES, 255, device)
    model.eval()
    total_eval_loss = 0.0
    num_batches = 0
    eval_loss_fn = OhemCrossEntropy()
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    # Sliding window parameters
    WINDOW_SIZE = 1024
    STRIDE = 512  # Overlap of 512 pixels

    with torch.no_grad():
        for batch in progress_bar:
            # Original images are [B, 3, 1024, 2048]
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            B, C, H, W = pixel_values.shape
            # Initialize accumulator for logits and a count mask for averaging
            # Logits from Segformer are 1/4 of input size
            full_logits = torch.zeros((B, N_CLASSES_CITYSCAPES, H // 4, W // 4), device=device)
            count_mask = torch.zeros((B, 1, H // 4, W // 4), device=device)

            # Slide across the width (0 to 2048)
            for x in range(0, W - WINDOW_SIZE + 1, STRIDE):
                window_input = pixel_values[:, :, :, x:x + WINDOW_SIZE]  # [B, 3, 1024, 1024]

                outputs = model(pixel_values=window_input)
                logits = outputs.logits  # [B, 19, 256, 256]

                # Map back to the full logit scale (1/4 of original)
                lx = x // 4
                lw = WINDOW_SIZE // 4
                full_logits[:, :, :, lx:lx + lw] += logits
                count_mask[:, :, :, lx:lx + lw] += 1

            # Average overlapping areas
            full_logits /= count_mask

            # Upsample to full resolution (1024x2048)
            upsampled_logits = F.interpolate(
                full_logits,
                size=labels.shape[-2:],
                mode="bicubic",
                align_corners=False
            ).softmax(dim=1)

            # Loss and Metrics
            eval_loss = loss_func(eval_loss_fn, upsampled_logits, labels, previous_cm)
            total_eval_loss += eval_loss.item()
            metrics.update(upsampled_logits, labels)

            predicted_masks = upsampled_logits.argmax(dim=1)
            metric.add_batch(
                predictions=predicted_masks.detach().cpu().numpy(),
                references=labels.detach().cpu().numpy()
            )
            num_batches += 1
            # Check at the end of every batch
            clear_cache_if_full(max_gb=3)

    # ... (rest of your return logic remains the same)
    avg_eval_loss = total_eval_loss / num_batches if num_batches > 0 else 0.0
    ious, miou, CM = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()

    return miou, ious, CM, macc, acc, mf1, f1, avg_eval_loss
'''


def evaluate_model_sw(model, dataloader, device, criterion, cm_weights):
    """
    Sliding Window Evaluation with Memory Optimization.
    """
    n_classes = 19
    metrics = Metrics(n_classes, 255, device)
    model.eval()

    total_eval_loss = 0.0
    num_batches = 0

    # Sliding window parameters
    WINDOW_SIZE = 1024
    STRIDE = 512

    # Create a 2D Gaussian/Linear window to smooth edges between overlaps
    # This prevents edge artifacts where windows meet
    window_weight = torch.ones((1, 1, 1024, 1024), device=device)
    # Optional: Apply a linear ramp to edges
    ramp = torch.linspace(0, 1, 128, device=device)
    window_weight[:, :, :128, :] *= ramp.view(1, 1, 128, 1)  # Top
    window_weight[:, :, -128:, :] *= ramp.flip(0).view(1, 1, 128, 1)  # Bottom

    progress_bar = tqdm(dataloader, desc="Evaluating (SW)", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device)  # [B, 3, 1024, 2048]
            labels = batch["labels"].to(device)
            B, C, H, W = pixel_values.shape

            # Accumulators (Keep on CPU if memory is tight, but GPU is faster)
            # We accumulate in 1/4 resolution because Segformer outputs 1/4 resolution
            feat_h, feat_w = H // 4, W // 4
            win_h, win_w = WINDOW_SIZE // 4, WINDOW_SIZE // 4
            stride_f = STRIDE // 4

            full_logits = torch.zeros((B, n_classes, feat_h, feat_w), device=device)
            count_mask = torch.zeros((B, 1, feat_h, feat_w), device=device)

            # Slide across the width
            for x in range(0, W - WINDOW_SIZE + 1, STRIDE):
                window_input = pixel_values[:, :, :, x:x + WINDOW_SIZE]

                with torch.cuda.amp.autocast():  # Use mixed precision for speed
                    outputs = model(pixel_values=window_input)
                    logits = outputs.logits  # [B, 19, 256, 256]

                lx = x // 4
                # Apply the window weight to the logits
                full_logits[:, :, :, lx:lx + win_w] += logits * window_weight[:, :, ::4, ::4]
                count_mask[:, :, :, lx:lx + win_w] += window_weight[:, :, ::4, ::4]

            # Final normalization
            full_logits /= (count_mask + 1e-6)

            # Upsample to full resolution for Loss and Metrics
            upsampled_logits = F.interpolate(
                full_logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            # Loss calculation using the new OO Class
            # Note: CAL expects probabilities, OHEM expects logits
            if isinstance(criterion, CALoss):
                loss = criterion(upsampled_logits.softmax(dim=1), labels, cm_weights)
            else:
                loss = criterion(upsampled_logits, labels)

            total_eval_loss += loss.item()

            # Metrics Update
            metrics.update(upsampled_logits.softmax(dim=1), labels)
            num_batches += 1

            # Periodic Cache Clear
            if num_batches % 10 == 0:
                torch.cuda.empty_cache()

    # Final Compute
    ious, miou, CM = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    avg_eval_loss = total_eval_loss / num_batches

    return miou, ious, CM, macc, acc, mf1, f1, avg_eval_loss


# ----------------------------------------------------------------------------------------------------------------------
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Validation and type casting for safety
    config['N_CLASSES_CITYSCAPES'] = int(config.get('N_CLASSES_CITYSCAPES', 19))
    config['NUM_EPOCHS'] = int(config.get('NUM_EPOCHS', 500))
    config['BATCH_SIZE'] = int(config.get('BATCH_SIZE', 8))

    return config

def log_config_to_file(log_file_path, config):
    """Logs the training configuration to the top of the CSV log file."""

    # Read the existing content
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        content = ""

    # Check if config is already logged
    if content.startswith("# CONFIG START"):
        return  # Configuration already logged

    # Convert config to a multi-line string with comments
    config_lines = ["# CONFIG START"]
    for k, v in config.items():
        config_lines.append(f"# {k}: {v}")
    config_lines.append("# CONFIG END\n")

    # Read the header line from the existing content (if it exists)
    header_line = ""
    lines = content.split('\n')
    for line in lines:
        if line.startswith("Epoch\t"):
            header_line = line
            break

    # If header exists, strip everything before it for clean rewrite
    if header_line:
        # Keep only lines from header onwards
        content_after_header = '\n'.join(lines[lines.index(header_line):])
    else:
        # If no header, keep the current content
        content_after_header = content

    # Prepend config to the rest of the content
    new_content = "\n".join(config_lines) + content_after_header

    # Write the new content
    with open(log_file_path, 'w') as f:
        f.write(new_content)

# ----------------------------------------------------------------------------------
def log_epoch_record(log_file_path, epoch, avg_train_loss, avg_eval_loss, epoch_time_seconds, optimizer, current_miou, ious, macc, mf1):
    # 1. Ensure all mean/overall metrics are converted to scalar types (float/item)
    current_miou_scalar = current_miou.item() if isinstance(current_miou,
                                                                (np.ndarray, torch.Tensor)) else current_miou
    macc_scalar = macc.item() if isinstance(macc, (np.ndarray, torch.Tensor)) else macc
    # acc_scalar (Overall Pixel Accuracy) OMITTED, as requested.
    mf1_scalar = mf1.item() if isinstance(mf1, (np.ndarray, torch.Tensor)) else mf1

    # 2. Convert the list of per-class ious ('ious' list) into a single string.
    ious_list_string = ",".join([f"{x:.4f}" for x in np.asarray(ious).flatten()])
    ious_list = np.asarray(ious).flatten().tolist()
    ious_formatted_list = [f"{x:.4f}" for x in ious_list]   # Define the new row data (dictionary)
    # Define the new row data
    new_row = {
        'Epoch': epoch + 1,
        'Train_Loss': f"{avg_train_loss:.4f}",
        'Eval_Loss': f"{avg_eval_loss:.4f}",
        'mIoU': f"{current_miou_scalar:.4f}",
        'mAcc': f"{macc_scalar:.4f}",
        'AvgF1': f"{mf1_scalar:.4f}",
        'Time_s': f"{epoch_time_seconds:.2f}",
        'LR': f"{optimizer.param_groups[0]['lr']:.6e}"
    }

    # Dynamically add the 19 separate class ious columns
    for i, ious_val_str in enumerate(ious_formatted_list):
        new_row[f'Class_iou_{i}'] = ious_val_str

    # Create a DataFrame for the new row
    df_new_row = pd.DataFrame([new_row])

    # Check if the file exists to decide whether to write the header
    write_header = not log_file_path.exists()

    # Append the new row to the CSV file
    df_new_row.to_csv(
        log_file_path,
        mode='a',  # Append mode
        header=write_header,  # Write header only on the first run
        index=False,  # Do not write the DataFrame index
        sep='\t'  # Use a tab delimiter to maintain your preferred structure
    )

    # --- End Pandas Logging Block ---

def show_pixel_values(pixels_batch):
    # 1. Move to CPU, denormalize, and convert to numpy
    # (Assuming img is already: pixel_values[0].cpu() * std + mean)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_np = (pixels_batch[0].cpu() * std + mean).numpy()

    # 2. Transpose from (C, H, W) -> (H, W, C)
    img_np = np.transpose(img_np, (1, 2, 0))

    # 3. Clip values to [0, 1] range to avoid overflow/artifacts
    # and scale to 0-255 for uint8 conversion
    img_np = np.clip(img_np, 0, 1)
    img_final = (img_np * 255).astype(np.uint8)

    # 4. View on monitor
    plt.imshow(img_final)
    plt.axis('off')  # Hide axes
    plt.show()
    return


def clear_cache_if_full(max_gb=3):
    # Standard HF cache location
    cache_path = Path.home() / ".cache/huggingface"

    if not cache_path.exists():
        return

    # Calculate size in bytes
    total_size = sum(f.stat().st_size for f in cache_path.glob('**/*') if f.is_file())
    size_gb = total_size / (1024 ** 3)

    if size_gb > max_gb:
        print(f"\n[Cache Alert] Cache size is {size_gb:.2f} GB. Clearing to save quota...")
        try:
            # We target 'datasets' specifically as it's usually the part that 'explodes'
            # Deleting the 'hub' folder might force a redownload of the model weights
            # dataset_cache = cache_path / "datasets"
            if cache_path.exists():
                shutil.rmtree(cache_path)

            # Clear temp files
            #temp_cache = cache_path / "temp"
            #if temp_cache.exists():
            #    shutil.rmtree(temp_cache)

            print("[Cache Alert] Cache cleared successfully.")
        except Exception as e:
            print(f"Error clearing cache: {e}")

'''
def Calc_Weights(confmat, confusion_type):
    n_classes = confmat.shape[0]
    if confusion_type == 'fn':
        TP_FN_FP = confmat
    elif confusion_type == 'fn_fp':
        TP_FN_FP = confmat + confmat.T - confmat * np.eye(n_classes, dtype=float)
    else:
        print('only fn or fn_fp')
        exit(0)

    sums = np.sum(TP_FN_FP, axis=1, keepdims=True)
    mask = sums > 0
    cm_weights = np.zeros_like(TP_FN_FP, dtype=float)
    cm_weights[mask.squeeze()] = TP_FN_FP[mask.squeeze()] / sums[mask.squeeze()]

    return cm_weights
'''

def Calc_Weights(confmat, confusion_type):
    n_classes = confmat.shape[0]
    if confusion_type == 'fn':
        # False Negatives focus
        TP_FN_FP = confmat.copy()
    elif confusion_type == 'fn_fp':
        # Symmetric Confusion (Both directions)
        TP_FN_FP = confmat + confmat.T
        np.fill_diagonal(TP_FN_FP, np.diag(confmat))  # Keep diagonal as is

    # Normalize rows
    row_sums = TP_FN_FP.sum(axis=1, keepdims=True)
    return np.divide(TP_FN_FP, row_sums, out=np.zeros_like(TP_FN_FP), where=row_sums != 0)

# ----------------------------------------------------------------------------------------------------------------------
def main():

    datasets.disable_caching()

    print("Current path = ", os.getcwd())

    parser = argparse.ArgumentParser(description="SegFormer Cityscapes Training")
    parser.add_argument("config_file", type=str, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    CITYSCAPES_CLASS_NAMES = [
        "road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light",
        "traffic_sign", "vegetation", "terrain", "sky", "person", "rider",
        "car", "truck", "bus", "train", "motorcycle", "bicycle"
    ]

    # --- Load Configuration ---
    config = load_config(args.config_file)
    n_classes_cityscapes = len(CITYSCAPES_CLASS_NAMES)
    num_epochs = config['NUM_EPOCHS']
    batch_size = config['BATCH_SIZE']
    base_model_name = config['BASE_MODEL_NAME']
    experiment_name = config['EXPERIMENT_NAME']
    data_set_path = config['DATA_SET_PATH']
    loss_type = config['LOSS_TYPE']
    confusion_type = config['CONFUSION_TYPE']
    # --- Training Configuration ---

    if loss_type == "cal_ohem":
        criterion = CALoss(n_classes=19)
    elif loss_type == "ohem":
        criterion = OHEMCrossEntropy()
    else:
        print("Invalid loss type")
        exit(1)

    experiment_dir = Path("./experiments") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Define file paths RELATIVE to the experiment_dir
    log_file_path = experiment_dir / "training_log.csv"
    checkpoint_path = experiment_dir / "last_checkpoint.pt"
    best_model_path = experiment_dir / "best_model.pt"

    # --- Logging Setup ---
    print(f"Logging configuration to {log_file_path}")
    log_config_to_file(log_file_path, config)

    # base_header = "Epoch\tTrain_Loss\tEval_Loss\tmIoU\tmAcc\tAvgF1\tTime_s"
    base_header = "Epoch\tTrain_Loss\tEval_Loss\tmIoU\tmAcc\tAvgF1\tTime_s\tLR"
    class_headers = "\t".join([f"{name}" for name in CITYSCAPES_CLASS_NAMES])
    header_template = f"{base_header}\t{class_headers}\n"

    write_header = not log_file_path.exists()

    if write_header:
        # If the file didn't exist, it was created by log_config_to_file, which we will now append the header to.
        with open(log_file_path, 'a') as f:
            f.write(header_template)
        print(f"Created new log file and appended header at {log_file_path}")
    else:
        # If the file exists, we check if the header is present (to avoid writing it again)
        try:
            with open(log_file_path, 'r') as f:
                content = f.read()
                if base_header.strip() not in content:
                    with open(log_file_path, 'a') as f_append:
                        f_append.write(header_template)
        except Exception:
            pass  # Ignore if file access fails

        print(f"Appending results to existing log file at {log_file_path}")

    # 1. Load the configuration and Model
    config = SegformerConfig.from_pretrained(base_model_name)
    config.num_labels = n_classes_cityscapes
    model = SegformerForSemanticSegmentation(config)

    # 2. Load the processor
    processor = SegformerImageProcessor.from_pretrained(base_model_name)
    processor.do_resize = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_epoch = 0
    best_miou = 0.0
    CM = np.eye(n_classes_cityscapes, dtype=float)  # Initial CM for CF Loss

    # 5. Define the OHEM loss function and optimizer/scheduler (Initialized BEFORE loading states)
    ohem_loss_fn = OhemCrossEntropy()
    initial_lr = 0.00001  # SegFormer usually uses 6e-5, adjusted here for typical AdamW usage
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    # scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scheduler = None  # Initialize to None for now

    metric = evaluate.load("mean_iou")

    # 4. Instantiate datasets and DataLoaders
    train_dataset = CityscapesDataset(root_dir=data_set_path, processor=processor, split="train", fine_annotation=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = CityscapesDataset(root_dir=data_set_path, processor=processor, split="val", fine_annotation=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    warmup_iters = 0
    decay_power = 0.9
    total_steps = len(train_dataloader) * num_epochs

    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_iters,
        num_training_steps=total_steps,
        lr_end=0.0,
        power=decay_power
    )

    # 3. MANUALLY LOAD PRE-TRAINED WEIGHTS & CHECKPOINT STATE
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 2. Manually move internal optimizer tensors (momentum/exp_avg/exp_avg_sq) to the GPU
        # This loop iterates through the optimizer's state dictionary and moves any tensor found to the device.
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch']
        best_miou = checkpoint['best_miou']
        CM = checkpoint['CM']
        scheduler.T_max = num_epochs
        print(f"Resuming from Epoch {start_epoch + 1} with Best mIoU: {best_miou:.4f}")
    else:
        try:

            #results = evaluate_cityscapes("nvidia/segformer-b0-finetuned-cityscapes-1024-1024", val_dataloader)
            #print(f"Mean IoU: {results['mean_iou']:.4f}")
            # Initial load for a new run (ImageNet weights)
            state_dict = SegformerForSemanticSegmentation.from_pretrained(base_model_name).state_dict()
            model.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded fine-tuned weights for {base_model_name}.")
            CM = np.eye(n_classes_cityscapes, dtype=float)
        except Exception as e:
            print(f"Warning: Failed to load state_dict directly. The head may be random. Error: {e}")
            print(f"No checkpoint found. Starting fresh from Epoch 1.")

    model.to(device)

    CM_Weight = Calc_Weights(CM, confusion_type)

    miou, ious, CM, macc, acc, mf1, f1, avg_loss = evaluate_model_sw(
        model,
        val_dataloader,
        device,
        criterion,
        CM_Weight
    )
    print(f"  Current mIoU: {miou:.4f}")

    if not checkpoint_path.exists():
        log_epoch_record(log_file_path, 0, 0, avg_loss, 0, optimizer,
                     miou, ious, macc, mf1)

    ema_cm = CM.copy()
    alpha = 0.5

    # 7. Training and Evaluation Loop
    for epoch in range(start_epoch, num_epochs):
        t_start_epoch = time.time()
        ema_cm = alpha * ema_cm + (1 - alpha) * CM

        CM_Weight = Calc_Weights(ema_cm, confusion_type)
        # Calculate CM_Weight for the current epoch based on CM from the *previous* evaluation
        '''
        if confusion_type == 'fn':
            TP_FN_FP = ema_cm
        elif confusion_type == 'fn_fp':
            TP_FN_FP = ema_cm + ema_cm.T - ema_cm * np.eye(n_classes_cityscapes, dtype=float)
        else:
            print('only fn or fn_fp')
            exit(0)

        sums = np.sum(TP_FN_FP, axis=1, keepdims=True)
        mask = sums > 0
        CM_Weight = np.zeros_like(TP_FN_FP, dtype=float)
        CM_Weight[mask.squeeze()] = TP_FN_FP[mask.squeeze()] / sums[mask.squeeze()]
        '''
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{num_epochs}: Starting Learning Rate (LR): {current_lr:.8f}")

        # ----------------------------------------------------------------------------------
        # --- TRAINING PHASE ---
        # ----------------------------------------------------------------------------------
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0
        avg_data_time = 0.0

        progress_bar_train = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs} (Train)",
            ncols=150
        )

        t_batch_start = time.time()

        for i, batch in enumerate(progress_bar_train):

            data_load_time = time.time() - t_batch_start
            if num_train_batches == 0:
                avg_data_time = data_load_time
            else:
                avg_data_time = avg_data_time * 0.9 + data_load_time * 0.1

            optimizer.zero_grad()
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # show_pixel_values(pixel_values)

            outputs = model(pixel_values=pixel_values)
            logits = F.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear")

            '''
            upsampled_logits = F.interpolate(
                outputs.logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False
            ).softmax(dim=1)
            '''

            # loss = CF_loss(loss_fn, upsampled_logits, labels, CM_Weight)
            # loss = RLoss(ohem_loss_fn, upsampled_logits, labels, CM_Weight)
            # If using CAL, pass probs; if using OHEM, pass logits
            if isinstance(criterion, CALoss):
                loss = criterion(logits.softmax(dim=1), labels, CM_Weight)
            else:
                loss = criterion(logits, labels)

            total_train_loss += loss.item()
            num_train_batches += 1

            progress_bar_train.set_postfix({
                "train_loss": f"{loss.item():.4f}",
                "data_time(avg)": f"{avg_data_time:.4f}s",
                "lr": f"{optimizer.param_groups[0]['lr']:.6e}"
            })

            loss.backward()
            optimizer.step()
            scheduler.step()
            t_batch_start = time.time()

            # Check at the end of every batch
            clear_cache_if_full(max_gb=3)

        # ----------------------------------------------------------------------------------
        # --- EVALUATION AND LOGGING ---
        # ----------------------------------------------------------------------------------

        t_end_epoch = time.time()
        epoch_time_seconds = t_end_epoch - t_start_epoch
        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0.0

        print(f"\nEpoch {epoch + 1}/{num_epochs} finished. Evaluating...")

        current_miou, ious, CM, macc, acc, mf1, f1, valid_loss = evaluate_model_sw(
            model,
            val_dataloader,
            device,
            criterion,
            CM_Weight
        )
        print(f"Epoch {epoch + 1} Report:")
        print(f"  Avg Train Loss: {avg_train_loss:.4f}")
        print(f"  Avg Eval Loss: {valid_loss:.4f}")
        print(f"  Current mIoU: {current_miou:.4f}")

        # ----------------------------------------------------------------------------------
        # --- LOGGING TO FILE (SCALARS + LIST - training_log.txt) ---
        # ----------------------------------------------------------------------------------
        log_epoch_record(log_file_path, epoch, avg_train_loss, valid_loss, epoch_time_seconds, optimizer,
                         current_miou, ious, macc, mf1)


        # ... (Rest of checkpointing and saving logic remains the same) ...
        # ... rest of the code ...

        # CHECKPOINTING (path now includes the experiment folder)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_miou': best_miou,
            'CM': CM.copy(),
        }, checkpoint_path)

        # Saving best model (path now includes the experiment folder)
        if current_miou > best_miou:
            best_miou = current_miou
            shutil.copy(checkpoint_path, best_model_path)
            print(f"New best mIoU: {best_miou:.4f}. Saving model to {best_model_path}")
            # model.save_pretrained(best_model_path)
            # processor.save_pretrained(best_model_path)
        else:
            print("mIoU did not improve. Not saving model.")

        # Check at the end of every epoch
        clear_cache_if_full(max_gb=3)
        print(f"Epoch {epoch + 1} finished. mIoU: {current_miou:.4f}")

if __name__ == '__main__':
    main()
    '''
    NUM_EPOCHS = 500
    model = torch.nn.Conv2d(3, 16, 3, 1, 1)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)

    max_iter = 500
    #sched = WarmupPolyLR(optim, power=0.9, max_iter=max_iter, warmup_iter=200, warmup_ratio=0.1, warmup='exp',
    #                     last_epoch=-1)

    sched = get_scheduler("", optim, NUM_EPOCHS, 2, 0, 1, 0.15)

    lrs = []

    for _ in range(max_iter):
        lr = sched.get_lr()[0]
        lrs.append(lr)
        optim.step()
        sched.step()

    import matplotlib.pyplot as plt
    import numpy as np

    plt.plot(np.arange(len(lrs)), np.array(lrs))
    plt.grid()
    plt.show()
    '''

