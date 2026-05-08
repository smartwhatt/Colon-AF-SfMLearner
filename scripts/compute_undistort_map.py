"""
compute_undistort_map.py
------------------------
Computes and saves omnidirectional→perspective undistortion maps for C3VDv2.

C3VDv2 uses an Olympus CF-HQ190L colonoscope calibrated with the Scaramuzza
omnidirectional camera model. The calibration parameters are in camera_intrinsics.txt
downloaded from the C3VDv2 dataset page.

Scaramuzza model parameters (from camera_intrinsics.txt):
    width, height    : image dimensions
    cx, cy           : principal point
    a0..a4           : polynomial coefficients (world→pixel mapping)
    c, d, e          : affine transformation coefficients (usually c≈1, d≈e≈0)

Usage:
    python compute_undistort_map.py \
        --intrinsics /path/to/camera_intrinsics.txt \
        --output_dir /path/to/save/maps \
        --out_width 512 --out_height 512 --out_fov 120

After running this once, the saved maps are reused by C3VDv2Dataset for fast loading.
"""

import argparse
import os
import numpy as np


def parse_intrinsics(txt_path: str) -> dict:
    """
    Parse C3VDv2 camera_intrinsics.txt.
    Expected format (key: value pairs, one per line):
        width: 1350
        height: 1080
        cx: 675.0
        cy: 540.0
        a0: -274.5
        a1: 0.0
        a2: 5.1e-4
        a3: -1.2e-7
        a4: 1.8e-10
        c: 1.0
        d: 0.0
        e: 0.0
    Adjust to match the actual file format in your download.
    """
    params = {}
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue
            key, val = line.split("=")
            params[key.strip()] = float(val.strip())
    return params


def scaramuzza_cam2world_z(rho: np.ndarray, params: dict) -> np.ndarray:
    """
    Cam-to-world polynomial: maps image radius rho (pixels from principal point)
    to the unnormalised z-component of the 3D direction vector.

    z = a0 + a1*rho + a2*rho^2 + a3*rho^3 + a4*rho^4

    The full (unnormalised) direction for an image point at radius rho and
    azimuth phi is  (rho*cos(phi), rho*sin(phi), z).
    """
    a0 = params["a0"]
    a1 = params.get("a1", 0.0)
    a2 = params.get("a2", 0.0)
    a3 = params.get("a3", 0.0)
    a4 = params.get("a4", 0.0)
    return a0 + a1 * rho + a2 * rho**2 + a3 * rho**3 + a4 * rho**4


def _build_theta_to_rho_lut(params: dict, max_rho: float = 700.0, n: int = 20000):
    """
    Build a lookup table that inverts the cam-to-world polynomial:
      given an incident angle theta, return the corresponding image radius rho.

    Strategy: sample rho densely, compute theta = arctan2(rho, z(rho)),
    then use np.interp for the inverse mapping.
    """
    rho_samples = np.linspace(0.0, max_rho, n)
    z_samples = scaramuzza_cam2world_z(rho_samples, params)
    # theta = angle from optical axis; z must be positive for the forward hemisphere
    theta_samples = np.arctan2(rho_samples, np.maximum(z_samples, 1e-6))
    return theta_samples, rho_samples


def compute_undistort_maps(
    params: dict,
    out_w: int,
    out_h: int,
    out_fov_deg: float,
) -> tuple:
    """
    Build OpenCV-compatible remap tables (map1, map2) that convert a virtual
    perspective image of size (out_w x out_h) with field-of-view out_fov_deg
    back to source omnidirectional pixel coordinates.

    The Scaramuzza a0..a4 coefficients are treated as the cam-to-world
    polynomial (image radius → unnormalised z), which is the standard output
    of OCamCalib.  The world-to-cam mapping is obtained by numerical inversion.

    Returns:
        map1, map2: float32 arrays of shape (out_h, out_w) — source x,y coords
        K_out: 3x3 intrinsic matrix of the virtual perspective camera
    """
    cx_src = params["cx"]
    cy_src = params["cy"]
    c = params.get("c", 1.0)
    d = params.get("d", 0.0)
    e = params.get("e", 0.0)

    # Virtual perspective camera intrinsics
    f_out = (out_w / 2.0) / np.tan(np.deg2rad(out_fov_deg / 2.0))
    cx_out = out_w / 2.0
    cy_out = out_h / 2.0
    K_out = np.array(
        [[f_out, 0, cx_out], [0, f_out, cy_out], [0, 0, 1]], dtype=np.float32
    )

    # Build grid of output pixel coordinates
    uu, vv = np.meshgrid(
        np.arange(out_w, dtype=np.float32), np.arange(out_h, dtype=np.float32)
    )

    # Direction vectors in camera coordinates (z forward)
    x_c = (uu - cx_out) / f_out
    y_c = (vv - cy_out) / f_out

    # Incident angle from optical axis
    r_3d = np.sqrt(x_c**2 + y_c**2)
    theta = np.arctan2(r_3d, np.ones_like(r_3d))  # z_c = 1

    # Invert cam-to-world polynomial: theta → rho (image radius in source)
    theta_lut, rho_lut = _build_theta_to_rho_lut(params)
    r_src = np.interp(theta, theta_lut, rho_lut).astype(np.float32)

    # Azimuthal direction in source image plane
    # At the optical centre (r_3d=0) phi is undefined; r_src=0 so it doesn't matter.
    phi = np.arctan2(y_c, x_c)
    x_norm = np.where(r_3d > 0, np.cos(phi), 0.0)
    y_norm = np.where(r_3d > 0, np.sin(phi), 0.0)

    # Apply affine stretch (c, d, e parameters) and shift to principal point
    x_src = r_src * (x_norm * c + y_norm * d) + cx_src
    y_src = r_src * (x_norm * e + y_norm) + cy_src

    map1 = x_src.astype(np.float32)
    map2 = y_src.astype(np.float32)

    return map1, map2, K_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--intrinsics", required=True, help="Path to camera_intrinsics.txt from C3VDv2"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save undistort_map1.npy / map2.npy / K_out.npy",
    )
    parser.add_argument("--out_width", type=int, default=512)
    parser.add_argument("--out_height", type=int, default=512)
    parser.add_argument(
        "--out_fov",
        type=float,
        default=120.0,
        help="Horizontal FOV of the virtual perspective camera (degrees)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Parsing intrinsics...")
    params = parse_intrinsics(args.intrinsics)
    print(f"  cx={params['cx']:.1f}, cy={params['cy']:.1f}")
    print(f"  a0={params['a0']:.4f}")

    print(
        f"Computing undistortion maps → {args.out_width}x{args.out_height}, FOV={args.out_fov}°..."
    )
    map1, map2, K_out = compute_undistort_maps(
        params, args.out_width, args.out_height, args.out_fov
    )

    np.save(os.path.join(args.output_dir, "undistort_map1.npy"), map1)
    np.save(os.path.join(args.output_dir, "undistort_map2.npy"), map2)
    np.save(os.path.join(args.output_dir, "K_out.npy"), K_out)

    print(f"Saved maps to {args.output_dir}")
    print(f"Virtual camera K:\n{K_out}")
    print(
        "\nNow update UNDISTORT_FX/FY/CX/CY in colon_datasets/c3vdv2_dataset.py with K_out values."
    )


if __name__ == "__main__":
    main()
