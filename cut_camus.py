#!/usr/bin/env python3
"""
Process 4-chamber CAMUS NIfTI sequences and save:

    • Whole LV mask  (blue)
    • Apical-slice mask (red)
    • Difference mask (blue minus red)
    • Original frame
    • NumPy arrays of the three masks

Usage
-----
python process_camus.py --src <IN_DIR> --dst <OUT_DIR> --angle 40
"""

from __future__ import annotations

import argparse
import math
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
import SimpleITK as sitk

# ----------------------------------------------------------------------------------------------------------------------
# GEOMETRY HELPERS
# ----------------------------------------------------------------------------------------------------------------------

def find_extreme_points(binary_img: np.ndarray) -> Tuple[Tuple[int, int], ...]:
    """
    Return (leftmost, rightmost, apex) extreme points of a binary contour image.
    """
    x, y = np.where(binary_img.T != 0)          # note transpose to treat (x,y)
    return (x[x.argmin()], y[y.argmin()]), (x[x.argmax()], y[y.argmax()]), (x[y.argmin()], y[y.argmin()])


def rotate_point(origin: Tuple[int, int],
                 pt: Tuple[int, int],
                 angle_rad: float) -> Tuple[int, int]:
    """
    Rotate *pt* around *origin* counter-clockwise by *angle_rad*.
    """
    ox, oy = origin
    px, py = pt
    qx = ox + math.cos(angle_rad) * (px - ox) - math.sin(angle_rad) * (py - oy)
    qy = oy + math.sin(angle_rad) * (px - ox) + math.cos(angle_rad) * (py - oy)
    return int(qx), int(qy)


def get_rotated_limits(apex: Tuple[int, int],
                       left: Tuple[int, int],
                       right: Tuple[int, int],
                       half_angle_deg: float) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Rotate left & right basal points symmetrically around the apex.
    """
    half_rad = math.radians(half_angle_deg)
    return rotate_point(apex, left, -half_rad), rotate_point(apex, right, +half_rad)


def pts_below_y(points: np.ndarray, y_ref: int) -> np.ndarray:
    """Return contour points that lie strictly below *y_ref*."""
    return points[points[:, 1] > y_ref]

# ----------------------------------------------------------------------------------------------------------------------
# DRAW / MASK HELPERS
# ----------------------------------------------------------------------------------------------------------------------

BLUE_BGR = (255, 0, 0)   # BGR (so this is blue)
RED_BGR  = (0, 0, 255)   # BGR (this is red)

def filled_mask(shape: Tuple[int, int], contour: np.ndarray) -> np.ndarray:
    """Return filled RGB mask with a white polygon inside *contour*."""
    m = np.zeros((*shape, 3), dtype=np.uint8)
    cv2.fillPoly(m, [contour], (255, 255, 255))
    return m

# ----------------------------------------------------------------------------------------------------------------------
# CORE PROCESSING
# ----------------------------------------------------------------------------------------------------------------------

def process_patient(patient_dir: Path,
                    dst_root : Path,
                    angle_deg: float) -> None:
    """
    Load the patient's half-sequence, iterate through frames, and save outputs.
    """
    try:
        nii_path = next(patient_dir.glob("*4CH_half_sequence.nii.gz"))
    except StopIteration:
        print(f"[WARN] {patient_dir.name}: no 4CH_half_sequence found")
        return

    volume = sitk.GetArrayFromImage(sitk.ReadImage(str(nii_path)))      # (T, H, W)

    for t, frame in enumerate(volume):
        process_frame(frame, dst_root / patient_dir.name, t, angle_deg)


def process_frame(frame: np.ndarray,
                  out_dir: Path,
                  t: int,
                  angle_deg: float) -> None:
    """
    Process a single frame and write PNG/NPY artefacts.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_u8 = frame.astype(np.uint8)

    # --- contour extraction ---------------------------------------------------
    _, thresh = cv2.threshold(frame_u8, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
    contour = contours[0].squeeze()                        # (N, 2)

    left, right, apex = find_extreme_points(frame_u8)
    left_rot, right_rot = get_rotated_limits(apex, left, right, angle_deg / 2)

    blue_contour = np.concatenate([np.array([apex, left, right]), contour])
    red_seg      = pts_below_y(contour, min(left_rot[1], right_rot[1]))
    red_contour  = np.concatenate([np.array([apex, left_rot, right_rot]), red_seg])

    blue_rgb = filled_mask(frame.shape, blue_contour)
    red_rgb  = filled_mask(frame.shape, red_contour)

    blue_mask = blue_rgb[:, :, 0] > 0
    red_mask  = red_rgb[:,  :, 0] > 0
    diff_mask = blue_mask & ~red_mask

    # multiply masks with original grayscale frame
    whole_seg = frame * blue_mask
    red_seg   = frame * red_mask
    diff_seg  = frame * diff_mask

    prefix = out_dir / f"{t:03d}"
    cv2.imwrite(str(prefix.with_suffix('_whole.png')), whole_seg)
    cv2.imwrite(str(prefix.with_suffix('_red.png')),   red_seg)
    cv2.imwrite(str(prefix.with_suffix('_diff.png')),  diff_seg)
    cv2.imwrite(str(prefix.with_suffix('_orig.png')),  frame)

    np.save(str(prefix.with_suffix('_mask_blue.npy')), blue_mask.astype(np.uint8))
    np.save(str(prefix.with_suffix('_mask_red.npy')),  red_mask.astype(np.uint8))
    np.save(str(prefix.with_suffix('_mask_diff.npy')), diff_mask.astype(np.uint8))

# ----------------------------------------------------------------------------------------------------------------------
# CLI + ENTRY POINT
# ----------------------------------------------------------------------------------------------------------------------

def iter_patients(root: Path) -> Iterable[Path]:
    return (p for p in root.iterdir() if p.is_dir())


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process CAMUS 4-chamber half sequences into LV masks."
    )
    parser.add_argument("--src",  required=True, type=Path,
                        help="Directory containing patient sub-folders with 4CH_half_sequence.nii.gz files")
    parser.add_argument("--dst",  required=True, type=Path,
                        help="Directory where processed outputs will be stored")
    parser.add_argument("--angle", required=True, type=float,
                        help="Included angle in degrees between the two cutting lines (e.g. 40)")
    return parser.parse_args()


def main() -> None:
    args = cli()
    for i, patient in enumerate(iter_patients(args.src), 1):
        print(f"[{i:3}] Processing {patient.name}")
        process_patient(patient, args.dst, args.angle)


if __name__ == "__main__":
    main()

