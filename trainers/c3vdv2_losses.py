"""
c3vdv2_losses.py
----------------
Loss functions for fine-tuning AF-SfMLearner on C3VDv2.

Three losses are combined:
  1. L_photo  — self-supervised photometric reprojection (original AF-SfM loss)
  2. L_depth  — supervised scale-invariant log depth loss against C3VDv2 GT
  3. L_pose   — supervised pose regression against C3VDv2 GT relative poses

Total loss:
    L = w_photo * L_photo + w_depth * L_depth + w_pose * L_pose

Recommended starting weights: w_photo=1.0, w_depth=0.5, w_pose=0.1
Tune w_depth upward if metric scale is drifting during evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import compute_pps_map


# ---------------------------------------------------------------------------
# 1. Supervised depth loss
# ---------------------------------------------------------------------------


def scale_invariant_log_loss(
    pred_depth: torch.Tensor, gt_depth: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Scale-invariant logarithmic depth loss (Eigen et al., 2014).
    Encourages the network to learn absolute metric depth.

    Args:
        pred_depth: (B, 1, H, W) predicted metric depth in metres
        gt_depth:   (B, 1, H, W) ground-truth metric depth in metres
        mask:       (B, 1, H, W) float mask — 1 = valid pixel, 0 = ignore
                    (automatically built from gt > 0 if None)
    Returns:
        Scalar loss.
    """
    if mask is None:
        mask = (gt_depth > 1e-3).float()

    # Also exclude saturated GT values (== 0.1 m means the TIFF was 65535)
    mask = mask * (gt_depth < 0.099).float()

    n = mask.sum().clamp(min=1)

    log_pred = torch.log(pred_depth.clamp(min=1e-4))
    log_gt = torch.log(gt_depth.clamp(min=1e-4))
    diff = (log_pred - log_gt) * mask

    # Scale-invariant component
    loss = (diff**2).sum() / n - 0.5 * (diff.sum() / n) ** 2
    return loss


def smooth_l1_depth_loss(
    pred_depth: torch.Tensor, gt_depth: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Alternative: smooth-L1 loss in log space. Less sensitive to outliers
    than MSE. Use this if training is unstable with scale_invariant_log_loss.
    """
    if mask is None:
        mask = (gt_depth > 1e-3).float() * (gt_depth < 0.099).float()

    log_pred = torch.log(pred_depth.clamp(min=1e-4))
    log_gt = torch.log(gt_depth.clamp(min=1e-4))
    loss = F.smooth_l1_loss(log_pred * mask, log_gt * mask, reduction="sum")
    return loss / mask.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# 2. Supervised pose loss
# ---------------------------------------------------------------------------


def pose_loss(
    pred_axang: torch.Tensor,
    pred_trans: torch.Tensor,
    gt_axang: torch.Tensor,
    gt_trans: torch.Tensor,
    rot_weight: float = 1.0,
    trans_weight: float = 1.0,
    log_s_rot: torch.Tensor = None,
    log_s_trans: torch.Tensor = None,
    ema_rot: torch.Tensor = None,
    ema_trans: torch.Tensor = None,
) -> torch.Tensor:
    """
    Supervised pose loss: L2 on axis-angle rotation + L2 on translation.

    When log_s_rot and log_s_trans are provided, uses the EMA-normalised
    uncertainty-weighted formulation:
        rot_n  = ||r_err||² / ema_rot
        trn_n  = ||t_err||² / ema_trans
        L = rot_n * exp(-log_s_rot)  + log_s_rot
          + trn_n * exp(-log_s_trans) + log_s_trans

    Division by the EMA keeps the effective loss near O(1), so the scalars
    converge from their zero init in a handful of steps instead of drifting
    deeply negative.

    Args:
        pred_axang:  (B, 3) predicted axis-angle from PoseNet
        pred_trans:  (B, 3) predicted translation from PoseNet
        gt_axang:    (B, 3) ground-truth axis-angle from C3VDv2 pose.txt
        gt_trans:    (B, 3) ground-truth translation in metres
        rot_weight:  scalar weight for rotation (used when log_s_rot is None)
        trans_weight: scalar weight for translation (used when log_s_trans is None)
        log_s_rot:   learnable log-variance for rotation (1,) tensor; None → fixed weights
        log_s_trans: learnable log-variance for translation (1,) tensor; None → fixed weights
        ema_rot:     running-mean of ||r_err||² (1,) tensor; None → no normalisation
        ema_trans:   running-mean of ||t_err||² (1,) tensor; None → no normalisation
    Returns:
        Scalar loss.
    """
    rot_loss = F.mse_loss(pred_axang, gt_axang)
    trans_loss = F.mse_loss(pred_trans, gt_trans)
    if log_s_rot is not None and log_s_trans is not None:
        s_r = log_s_rot.squeeze()
        s_t = log_s_trans.squeeze()
        if ema_rot is not None:
            rot_loss = rot_loss / ema_rot.squeeze().clamp(min=1e-8)
        if ema_trans is not None:
            trans_loss = trans_loss / ema_trans.squeeze().clamp(min=1e-8)
        return (
            rot_loss * torch.exp(-s_r) + s_r
            + trans_loss * torch.exp(-s_t) + s_t
        )
    return rot_weight * rot_loss + trans_weight * trans_loss


# ---------------------------------------------------------------------------
# 3. Photometric reprojection loss (AF-SfM compatible)
# ---------------------------------------------------------------------------


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Simplified SSIM for photometric loss (3-channel)."""
    C1, C2 = 0.01**2, 0.03**2
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sig_x = F.avg_pool2d(x**2, 3, 1, 1) - mu_x**2
    sig_y = F.avg_pool2d(y**2, 3, 1, 1) - mu_y**2
    sig_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sig_x + sig_y + C2)
    return num / den.clamp(min=1e-6)


def photometric_loss(
    target: torch.Tensor,
    reconstructed: torch.Tensor,
    occlusion_mask: torch.Tensor = None,
    ssim_weight: float = 0.85,
) -> torch.Tensor:
    """
    Photometric reprojection loss: weighted combination of L1 and SSIM.
    This mirrors the loss already used in AF-SfMLearner's trainer.

    Args:
        target:          (B, 3, H, W) original frame
        reconstructed:   (B, 3, H, W) frame warped from adjacent frame via predicted depth+pose
        occlusion_mask:  (B, 1, H, W) 1=occluded (exclude from loss), from C3VDv2 occlusions/
        ssim_weight:     weight for SSIM term (1-ssim_weight for L1)
    Returns:
        Scalar loss.
    """
    l1 = (target - reconstructed).abs()
    sim = (1 - ssim(target, reconstructed)) / 2.0  # 0=identical, 1=different

    loss_map = ssim_weight * sim + (1 - ssim_weight) * l1

    if occlusion_mask is not None:
        # Exclude occluded pixels — they violate the brightness constancy assumption
        valid = 1.0 - occlusion_mask  # 1 = valid, 0 = occluded
        return (loss_map * valid).sum() / valid.sum().clamp(min=1)
    else:
        return loss_map.mean()


# ---------------------------------------------------------------------------
# 4. Specular weak-supervision helpers
# ---------------------------------------------------------------------------


def compute_specular_pseudo_labels(
    image: torch.Tensor,
    threshold: float = 0.98,
) -> torch.Tensor:
    """Generate weak per-pixel specular labels from near-saturated pixels.

    Args:
        image:      (B, 3, H, W) float RGB in [0, 1]
        threshold:  intensity above which a pixel is considered specular

    Returns:
        (B, 1, H, W) float labels in {0, 1}
    """
    intensity = image.mean(dim=1, keepdim=True)  # (B,1,H,W)
    labels = (intensity >= threshold).float()
    # Dilate by 1 pixel via max-pool to cover specular halos
    labels = F.max_pool2d(labels, kernel_size=3, stride=1, padding=1)
    # Exclude very dark pixels (likely invalid/noise, not specular)
    valid = (intensity > 0.05).float()
    return labels * valid


def specular_bce_loss(
    pred_specular: torch.Tensor,
    pseudo_labels: torch.Tensor,
    valid_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Binary cross-entropy between the predicted specular map and pseudo-labels.

    Args:
        pred_specular:  (B, 1, H, W) sigmoid specular confidence in [0, 1]
        pseudo_labels:  (B, 1, H, W) float labels in {0, 1}
        valid_mask:     (B, 1, H, W) float mask — 1 = use, 0 = ignore

    Returns:
        Scalar loss.
    """
    loss_map = F.binary_cross_entropy(
        pred_specular, pseudo_labels, reduction="none"
    )
    if valid_mask is not None:
        n = valid_mask.sum().clamp(min=1)
        return (loss_map * valid_mask).sum() / n
    return loss_map.mean()


# ---------------------------------------------------------------------------
# 5. PPSNet-style supervised shading loss
# ---------------------------------------------------------------------------


def _masked_mean_per_image(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.flatten(1).sum(dim=1).clamp(min=1).view(-1, 1, 1, 1)
    return (value * mask).flatten(1).sum(dim=1).view(-1, 1, 1, 1) / denom


def pps_supervised_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    inv_K: torch.Tensor,
    valid_mask: torch.Tensor = None,
    light_position: torch.Tensor = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Supervised PPS consistency loss between predicted and GT depth.

    PPS maps are normalized per image over valid pixels so this term acts as a
    geometry regularizer rather than an absolute exposure or scale solver.
    """
    if valid_mask is None:
        valid_mask = (gt_depth > 1e-3).float()
    else:
        valid_mask = valid_mask.float()
    if valid_mask.shape[2:] != pred_depth.shape[2:]:
        valid_mask = F.interpolate(
            valid_mask, size=pred_depth.shape[2:], mode="nearest"
        )
    if gt_depth.shape[2:] != pred_depth.shape[2:]:
        gt_depth = F.interpolate(
            gt_depth, size=pred_depth.shape[2:], mode="nearest"
        )

    if valid_mask.sum().detach().item() <= 0:
        return pred_depth.sum() * 0.0
    if light_position is None:
        light_position = pred_depth.new_tensor([0.0015, 0.0, 0.0])

    pps_pred = compute_pps_map(pred_depth.clamp(min=eps), inv_K, light_position)
    with torch.no_grad():
        pps_gt = compute_pps_map(gt_depth.clamp(min=eps), inv_K, light_position)

    mean_pred = _masked_mean_per_image(pps_pred, valid_mask).clamp(min=eps)
    mean_gt = _masked_mean_per_image(pps_gt, valid_mask).clamp(min=eps)
    pps_pred = pps_pred / mean_pred
    pps_gt = pps_gt / mean_gt

    loss_map = F.smooth_l1_loss(pps_pred, pps_gt, reduction="none")
    return (loss_map * valid_mask).sum() / valid_mask.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------


class C3VDv2FineTuneLoss(nn.Module):
    """
    Combined loss for AF-SfMLearner fine-tuning on C3VDv2.

    Args:
        w_photo:  weight for photometric reprojection loss
        w_depth:  weight for supervised depth loss
        w_pose:   weight for supervised pose loss
        depth_loss_type: "scale_invariant" (default) or "smooth_l1"
    """

    def __init__(
        self,
        w_photo: float = 1.0,
        w_depth: float = 0.5,
        w_pose: float = 0.1,
        depth_loss_type: str = "scale_invariant",
    ):
        super().__init__()
        self.w_photo = w_photo
        self.w_depth = w_depth
        self.w_pose = w_pose
        self.depth_loss_fn = (
            scale_invariant_log_loss
            if depth_loss_type == "scale_invariant"
            else smooth_l1_depth_loss
        )

    def forward(
        self,
        # Photometric inputs
        color_curr: torch.Tensor,  # (B,3,H,W)
        recon_from_prev: torch.Tensor,  # (B,3,H,W) warped prev frame
        recon_from_next: torch.Tensor,  # (B,3,H,W) warped next frame
        occlusion: torch.Tensor,  # (B,1,H,W)
        # Depth inputs
        pred_depth: torch.Tensor,  # (B,1,H,W) predicted depth in metres
        gt_depth: torch.Tensor,  # (B,1,H,W) GT depth in metres
        # Pose inputs
        pred_axang_prev: torch.Tensor,  # (B,3)
        pred_trans_prev: torch.Tensor,  # (B,3)
        pred_axang_next: torch.Tensor,  # (B,3)
        pred_trans_next: torch.Tensor,  # (B,3)
        gt_axang_prev: torch.Tensor,  # (B,3)
        gt_trans_prev: torch.Tensor,  # (B,3)
        gt_axang_next: torch.Tensor,  # (B,3)
        gt_trans_next: torch.Tensor,  # (B,3)
    ) -> dict:
        """
        Returns a dict with individual and total losses for logging.
        """
        # ---- Photometric loss -------------------------------------------
        l_photo_prev = photometric_loss(color_curr, recon_from_prev, occlusion)
        l_photo_next = photometric_loss(color_curr, recon_from_next, occlusion)
        l_photo = (l_photo_prev + l_photo_next) / 2.0

        # ---- Depth loss --------------------------------------------------
        gt_mask = (gt_depth > 1e-3).float() * (gt_depth < 0.099).float()
        l_depth = self.depth_loss_fn(pred_depth, gt_depth, gt_mask)

        # ---- Pose loss ---------------------------------------------------
        l_pose_prev = pose_loss(
            pred_axang_prev, pred_trans_prev, gt_axang_prev, gt_trans_prev
        )
        l_pose_next = pose_loss(
            pred_axang_next, pred_trans_next, gt_axang_next, gt_trans_next
        )
        l_pose = (l_pose_prev + l_pose_next) / 2.0

        # ---- Total -------------------------------------------------------
        total = self.w_photo * l_photo + self.w_depth * l_depth + self.w_pose * l_pose

        return {
            "loss": total,
            "loss_photo": l_photo,
            "loss_depth": l_depth,
            "loss_pose": l_pose,
        }
