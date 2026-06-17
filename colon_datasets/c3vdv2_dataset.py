"""
C3VDv2 Dataset for AF-SfMLearner fine-tuning.

C3VDv2 file structure per sequence:
  <sequence_root>/
    rgb/           NNNN.png          (1350x1080, distorted omnidirectional)
    depth/         NNNN_depth.tiff   (16-bit, 0-65535 → 0-100mm)
    normals/       NNNN_normals.tiff (16-bit RGB)
    occlusions/    NNNN_occlusion.tiff
    optical_flow/  NNNN_flow.tiff
    diffuse/       NNNN_diffuse.png
    pose.txt       one 4x4 row-major camera-to-world matrix per line

Camera: Olympus CF-HQ190L omnidirectional fisheye.
We undistort to a virtual perspective crop before feeding to AF-SfMLearner.
"""

import os
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms


cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
# ---------------------------------------------------------------------------
# Omnidirectional → perspective undistortion helpers
# ---------------------------------------------------------------------------

# These are approximate perspective intrinsics after undistortion.
# Replace with values computed from camera_intrinsics.txt + your undistortion map.
# The undistorted crop is 512x512 centred on the optical axis.
UNDISTORT_WIDTH = 512
UNDISTORT_HEIGHT = 512
UNDISTORT_FX = 147.8  # pixels — tune after running compute_undistort_map.py
UNDISTORT_FY = 147.8
UNDISTORT_CX = 256.0
UNDISTORT_CY = 256.0

K_UNDISTORTED = np.array(
    [[UNDISTORT_FX, 0, UNDISTORT_CX], [0, UNDISTORT_FY, UNDISTORT_CY], [0, 0, 1]],
    dtype=np.float32,
)


def load_undistort_maps(map_path: str):
    """
    Load pre-computed undistortion maps (computed once by compute_undistort_map.py).
    Maps are saved as .npy files: map1 (x-coords) and map2 (y-coords).
    """
    map1 = np.load(os.path.join(map_path, "undistort_map1.npy"))
    map2 = np.load(os.path.join(map_path, "undistort_map2.npy"))
    return map1, map2


def apply_undistort(img_bgr: np.ndarray, map1, map2) -> np.ndarray:
    """Remap a raw omnidirectional frame to a perspective crop."""
    return cv2.remap(img_bgr, map1, map2, interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Depth helpers
# ---------------------------------------------------------------------------

DEPTH_MAX_MM = 100.0
DEPTH_MAX_M = DEPTH_MAX_MM / 1000.0  # 0.1 m


def decode_depth_tiff(path: str) -> np.ndarray:
    """
    Load a C3VDv2 16-bit depth TIFF and return metric depth in METRES.
    Pixel value 0 → 0 mm, pixel value 65535 → 100 mm.
    Invalid pixels (value == 0 or == 65535) are set to 0.
    """
    raw = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if raw is None:
        raise FileNotFoundError(f"Depth file not found: {path}")
    raw = raw.astype(np.float32)
    depth_m = raw / 65535.0 * DEPTH_MAX_M
    # Mask out invalid pixels
    depth_m[raw == 0] = 0.0
    depth_m[raw == 65535] = 0.0
    return depth_m


def undistort_depth(depth: np.ndarray, map1, map2) -> np.ndarray:
    """
    Remap a depth map using the same maps as RGB undistortion.
    Use INTER_NEAREST to avoid blending across depth discontinuities.
    """
    return cv2.remap(depth, map1, map2, interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------


def load_poses(pose_txt: str) -> list:
    """
    Load pose.txt and return a list of 4x4 numpy arrays (camera-to-world, row-major).
    """
    poses = []
    with open(pose_txt, "r") as f:
        for line in f:
            vals = list(map(float, line.strip().replace(',', ' ').split()))
            if len(vals) == 16:
                T = np.array(vals, dtype=np.float64).reshape(4, 4)
                poses.append(T)
    return poses


def relative_pose(T_cw_i: np.ndarray, T_cw_j: np.ndarray) -> np.ndarray:
    """
    Compute the relative pose T_ij = T_cw_i^{-1} @ T_cw_j
    (transforms points from frame j into frame i).
    """
    return np.linalg.inv(T_cw_i) @ T_cw_j


def pose_to_axisangle_translation(T: np.ndarray):
    """
    Decompose a 4x4 pose matrix into axis-angle (3,) and translation (3,).
    Used to supervise the PoseNet output.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    # Rodrigues formula
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten().astype(np.float32), t.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class C3VDv2Dataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for AF-SfMLearner fine-tuning on C3VDv2.

    Each sample returns a triplet (t-1, t, t+1) of undistorted RGB frames
    plus the ground-truth depth map and relative pose for the centre frame.

    Args:
        data_root:      Root directory containing all C3VDv2 sequences.
        split_file:     Path to a text file listing "<sequence_name> <frame_idx>" per line.
        undistort_maps: Path to directory containing undistort_map1.npy / undistort_map2.npy.
        target_size:    (H, W) to resize undistorted frames before feeding to the network.
        augment:        Whether to apply colour jitter augmentation.
    """

    MIN_DEPTH = 0.001  # 1 mm in metres  — below this is considered invalid
    MAX_DEPTH = 0.10  # 100 mm in metres — C3VDv2 maximum

    def __init__(
        self,
        data_root: str,
        split_file: str,
        undistort_maps: str,
        target_size: tuple = (256, 320),
        augment: bool = True,
    ):
        self.data_root = data_root
        self.target_size = target_size  # (H, W)
        self.augment = augment

        # Load undistortion maps
        self.map1, self.map2 = load_undistort_maps(undistort_maps)

        # Read split file: each line → (sequence_name, frame_index)
        self.samples = []
        with open(split_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    seq, idx = parts[0], int(parts[1])
                    self.samples.append((seq, idx))

        # Pre-load poses per sequence (cheap, avoids repeated IO)
        self._pose_cache = {}

        # Colour augmentation (only applied to RGB, never to depth)
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )
        self.to_tensor = transforms.ToTensor()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_poses(self, seq: str) -> list:
        if seq not in self._pose_cache:
            pose_file = os.path.join(self.data_root, seq, "pose.txt")
            self._pose_cache[seq] = load_poses(pose_file)
        return self._pose_cache[seq]

    def _load_rgb(self, seq: str, idx: int) -> np.ndarray:
        path = os.path.join(self.data_root, seq, "rgb", f"{idx:04d}.png")
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"RGB frame not found: {path}")
        img = apply_undistort(img, self.map1, self.map2)
        img = cv2.resize(img, (self.target_size[1], self.target_size[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img  # HxWx3 uint8

    def _load_depth(self, seq: str, idx: int) -> np.ndarray:
        path = os.path.join(self.data_root, seq, "depth", f"{idx:04d}_depth.tiff")
        depth = decode_depth_tiff(path)
        depth = undistort_depth(depth, self.map1, self.map2)
        depth = cv2.resize(
            depth,
            (self.target_size[1], self.target_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        return depth  # HxW float32 in metres

    def _load_occlusion(self, seq: str, idx: int) -> np.ndarray:
        base = os.path.join(self.data_root, seq, "occlusions", f"{idx:04d}_occlusion")
        path = next((p for p in (base + ".tiff", base + ".png") if os.path.exists(p)), None)
        if path is None:
            return None
        occ = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        occ = cv2.remap(occ, self.map1, self.map2, interpolation=cv2.INTER_NEAREST)
        occ = cv2.resize(
            occ,
            (self.target_size[1], self.target_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        return (occ == 255).astype(np.float32)  # binary mask

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, index: int) -> dict:
        seq, idx = self.samples[index]
        poses = self._get_poses(seq)
        n_frames = len(poses)

        # Clamp indices so we always have a valid triplet
        idx_prev = max(0, idx - 1)
        idx_curr = idx
        idx_next = min(n_frames - 1, idx + 1)

        # ---- RGB triplet ------------------------------------------------
        img_prev = self._load_rgb(seq, idx_prev)
        img_curr = self._load_rgb(seq, idx_curr)
        img_next = self._load_rgb(seq, idx_next)
        img_prev_aug = img_prev
        img_curr_aug = img_curr
        img_next_aug = img_next

        if self.augment:
            # Apply the SAME jitter to all three frames (consistency)
            fn_idx, brightness, contrast, saturation, hue = transforms.ColorJitter.get_params(
                self.color_jitter.brightness,
                self.color_jitter.contrast,
                self.color_jitter.saturation,
                self.color_jitter.hue,
            )
            jitter_fn = transforms.ColorJitter(
                brightness=(brightness, brightness),
                contrast=(contrast, contrast),
                saturation=(saturation, saturation),
                hue=(hue, hue),
            )
            img_prev_aug = np.array(jitter_fn(Image.fromarray(img_prev)))
            img_curr_aug = np.array(jitter_fn(Image.fromarray(img_curr)))
            img_next_aug = np.array(jitter_fn(Image.fromarray(img_next)))

        # Convert to float tensors [0, 1]
        to_t = lambda x: torch.from_numpy(x.transpose(2, 0, 1)).float() / 255.0
        t_prev = to_t(img_prev)
        t_curr = to_t(img_curr)
        t_next = to_t(img_next)
        t_prev_aug = to_t(img_prev_aug)
        t_curr_aug = to_t(img_curr_aug)
        t_next_aug = to_t(img_next_aug)

        # ---- Ground-truth depth -----------------------------------------
        depth = self._load_depth(seq, idx_curr)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()  # 1xHxW

        # ---- Occlusion mask (used to weight photometric loss) -----------
        occ = self._load_occlusion(seq, idx_curr)
        if occ is not None:
            occ_tensor = torch.from_numpy(occ).unsqueeze(0).float()  # 1xHxW
        else:
            occ_tensor = torch.zeros(1, *self.target_size)

        # ---- Relative poses (curr → prev, curr → next) ------------------
        T_prev = poses[idx_prev]
        T_curr = poses[idx_curr]
        T_next = poses[idx_next]

        T_curr2prev = relative_pose(T_curr, T_prev)
        T_curr2next = relative_pose(T_curr, T_next)

        axang_prev, trans_prev = pose_to_axisangle_translation(T_curr2prev)
        axang_next, trans_next = pose_to_axisangle_translation(T_curr2next)

        # ---- Camera intrinsics (normalised to [0,1] as AF-SfM expects) --
        # Normalise by the undistorted image dimensions (512×512), not the
        # target_size.  The trainer multiplies K_norm by (W//2^s, H//2^s) at
        # each scale to recover pixel-unit intrinsics; dividing by UNDISTORT
        # dimensions ensures that multiplication gives the correctly scaled K.
        K_norm = np.array(
            [
                [K_UNDISTORTED[0, 0] / UNDISTORT_WIDTH,  0, K_UNDISTORTED[0, 2] / UNDISTORT_WIDTH],
                [0, K_UNDISTORTED[1, 1] / UNDISTORT_HEIGHT, K_UNDISTORTED[1, 2] / UNDISTORT_HEIGHT],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        return {
            # RGB frames
            "color_curr": t_curr,  # 3xHxW
            "color_prev": t_prev,  # 3xHxW
            "color_next": t_next,  # 3xHxW
            "color_curr_aug": t_curr_aug,  # 3xHxW
            "color_prev_aug": t_prev_aug,  # 3xHxW
            "color_next_aug": t_next_aug,  # 3xHxW
            # Ground truth depth
            "gt_depth": depth_tensor,  # 1xHxW  (metres)
            # Occlusion mask (1 = occluded, exclude from photometric loss)
            "occlusion": occ_tensor,  # 1xHxW
            # Ground truth relative poses
            "axang_curr2prev": torch.from_numpy(axang_prev),  # (3,)
            "trans_curr2prev": torch.from_numpy(trans_prev),  # (3,)
            "axang_curr2next": torch.from_numpy(axang_next),  # (3,)
            "trans_curr2next": torch.from_numpy(trans_next),  # (3,)
            # Intrinsics
            "K": torch.from_numpy(K_norm),  # 3x3
            # Metadata
            "seq": seq,
            "frame_idx": idx_curr,
        }

    def __len__(self) -> int:
        return len(self.samples)
