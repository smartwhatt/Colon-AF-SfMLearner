"""Shared helpers for C3VDv2 source replay and C3VDv2 trainer batches."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from colon_datasets.c3vdv2_metrics import build_c3vdv2_valid_mask_torch
from .c3vdv2_losses import scale_invariant_log_loss, pose_loss


def adapt_c3vdv2_batch(batch, opt, device=None):
    """Map C3VDv2/HF batch keys to the tuple-key format used by trainers."""
    if device is None:
        device = next(
            (v.device for v in batch.values() if hasattr(v, "device")), None
        )
    out = {}

    for f_id, key in {0: "color_curr", -1: "color_prev", 1: "color_next"}.items():
        t = batch[key].to(device)
        aug = batch.get(f"{key}_aug", batch[key]).to(device)
        out[("color", f_id, 0)] = t
        out[("color_aug", f_id, 0)] = aug

    if "K" in batch:
        k_norm = batch["K"].to(device)
        batch_size = k_norm.shape[0]
        for scale in opt.scales:
            h_s = opt.height // (2 ** scale)
            w_s = opt.width // (2 ** scale)
            k_44 = torch.zeros(batch_size, 4, 4, device=device)
            k_44[:, :3, :3] = k_norm
            k_44[:, 0, :] *= w_s
            k_44[:, 1, :] *= h_s
            k_44[:, 3, 3] = 1.0
            out[("K", scale)] = k_44
            out[("inv_K", scale)] = torch.linalg.inv(k_44)

    for scale in opt.scales[1:]:
        h = opt.height // (2 ** scale)
        w = opt.width // (2 ** scale)
        for f_id in opt.frame_ids:
            if ("color", f_id, 0) in out:
                out[("color", f_id, scale)] = F.interpolate(
                    out[("color", f_id, 0)], [h, w],
                    mode="bilinear", align_corners=False,
                )

    skip = {
        "color_curr",
        "color_prev",
        "color_next",
        "color_curr_aug",
        "color_prev_aug",
        "color_next_aug",
        "K",
        "seq",
        "frame_idx",
    }
    for key, val in batch.items():
        if key not in skip:
            out[key] = val.to(device) if hasattr(val, "to") else val

    return out


def raw_log_depth_smooth_l1_loss(pred_depth, gt_depth, mask, eps=1e-6):
    """Metric-scale log-depth loss; unlike SILog, global scale is penalized."""
    if gt_depth.shape[2:] != pred_depth.shape[2:]:
        gt_depth = F.interpolate(
            gt_depth, size=pred_depth.shape[2:], mode="nearest"
        )
    if mask.shape[2:] != pred_depth.shape[2:]:
        mask = F.interpolate(mask.float(), size=pred_depth.shape[2:], mode="nearest")
    mask = mask.float()
    valid = mask.sum().clamp(min=1)
    pred_log = torch.log(pred_depth.clamp(min=eps))
    gt_log = torch.log(gt_depth.clamp(min=eps))
    loss_map = F.smooth_l1_loss(pred_log, gt_log, reduction="none")
    return (loss_map * mask).sum() / valid


def source_replay_depth_loss(pred_depth, gt_depth, occlusion, loss_type):
    mask = build_c3vdv2_valid_mask_torch(gt_depth, occlusion).float()
    if loss_type == "smooth_l1_log":
        return raw_log_depth_smooth_l1_loss(pred_depth, gt_depth, mask)
    if loss_type == "scale_invariant_log":
        return scale_invariant_log_loss(pred_depth, gt_depth, mask)
    raise ValueError(f"Unsupported source_replay_depth_loss={loss_type!r}")


def supervised_pose_loss_from_batch(inputs, outputs, frame_ids, balance=None):
    gt_key_map = {
        -1: ("axang_curr2prev", "trans_curr2prev"),
         1: ("axang_curr2next", "trans_curr2next"),
    }

    total_pose = 0.0
    n = 0
    for f_i in frame_ids[1:]:
        if f_i == "s" or f_i not in gt_key_map:
            continue
        axk, trk = gt_key_map[f_i]
        if axk not in inputs or ("axisangle", 0, f_i) not in outputs:
            continue
        pred_ax = outputs[("axisangle", 0, f_i)].squeeze(1)
        pred_tr = outputs[("translation", 0, f_i)].squeeze(1)
        total_pose += pose_loss(
            pred_ax, pred_tr, inputs[axk], inputs[trk], balance=balance
        )
        n += 1
    if n == 0:
        return None
    return total_pose / n
