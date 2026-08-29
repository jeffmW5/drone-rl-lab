"""Marker detector for the vision-hover "rise-to-level + distance-hold" task.

Detects a plain high-contrast square marker (e.g. a 1x1 inch black square
drawn on white paper) in a grayscale camera frame, via classical CV
(threshold -> contour -> bounding box). No fiducial library, no learned
model -- this is the "teacher" that self-labels captured frames; see
VISION_HOVER_PLAN.md's "Simplified first task" section.

Output is two numbers, matching the task's reduced control problem:
  - vertical_offset_px: marker center's vertical distance from frame center
    (negative = marker above center = drone should rise; positive = below)
  - size_px: the marker's apparent size in pixels (sqrt of bbox area),
    a proxy for distance -- see distance calibration (step 2, Windows-side)
    for the size <-> real-distance mapping.

Usage:
    python3 marker_detector.py --self-test
    python3 marker_detector.py path/to/frame.png
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class MarkerDetection:
    found: bool
    center_x: float = 0.0
    center_y: float = 0.0
    vertical_offset_px: float = 0.0
    horizontal_offset_px: float = 0.0
    size_px: float = 0.0
    bbox: Optional[tuple] = None  # (x, y, w, h)


def detect_marker(
    gray: np.ndarray,
    min_area_frac: float = 0.001,
    max_area_frac: float = 0.5,
    min_squareness: float = 0.6,
) -> MarkerDetection:
    """Detect a single high-contrast square marker in a grayscale frame.

    min_squareness: bbox aspect ratio must be within
        [min_squareness, 1/min_squareness] to be considered "square enough".
    """
    if gray.ndim != 2:
        raise ValueError("expected a single-channel grayscale image")

    h, w = gray.shape
    frame_area = h * w
    min_area = min_area_frac * frame_area
    max_area = max_area_frac * frame_area

    # Otsu threshold picks the split point automatically; marker is the
    # darker region on a lighter background (or vice versa), so try both
    # polarities and take whichever yields a better candidate.
    _, thresh_dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, thresh_light = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    best: Optional[MarkerDetection] = None
    best_area = 0.0

    for thresh in (thresh_dark, thresh_light):
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw == 0 or bh == 0:
                continue
            aspect = bw / bh
            squareness = min(aspect, 1.0 / aspect)
            if squareness < min_squareness:
                continue
            if area <= best_area:
                continue
            best_area = area
            cx = x + bw / 2.0
            cy = y + bh / 2.0
            best = MarkerDetection(
                found=True,
                center_x=cx,
                center_y=cy,
                vertical_offset_px=cy - h / 2.0,
                horizontal_offset_px=cx - w / 2.0,
                size_px=math.sqrt(bw * bh),
                bbox=(x, y, bw, bh),
            )

    return best if best is not None else MarkerDetection(found=False)


def _make_synthetic_frame(
    width: int = 324,
    height: int = 244,
    square_cx: float = 180.0,
    square_cy: float = 100.0,
    square_size: int = 40,
    bg: int = 230,
    fg: int = 20,
) -> np.ndarray:
    img = np.full((height, width), bg, dtype=np.uint8)
    x0 = int(square_cx - square_size / 2)
    y0 = int(square_cy - square_size / 2)
    img[y0 : y0 + square_size, x0 : x0 + square_size] = fg
    return img


def _self_test() -> bool:
    ok = True
    width, height = 324, 244

    cases = [
        # (name, square_cx, square_cy, square_size)
        ("centered", width / 2, height / 2, 40),
        ("above-center, small (far)", 150, 60, 16),
        ("below-center, large (near)", 200, 180, 80),
        ("off to the side", 40, 120, 30),
    ]

    for name, cx, cy, size in cases:
        frame = _make_synthetic_frame(width, height, cx, cy, size)
        det = detect_marker(frame)
        if not det.found:
            print(f"FAIL [{name}]: marker not detected")
            ok = False
            continue

        expected_vert = cy - height / 2.0
        expected_horiz = cx - width / 2.0
        expected_size = float(size)

        vert_err = abs(det.vertical_offset_px - expected_vert)
        horiz_err = abs(det.horizontal_offset_px - expected_horiz)
        size_err = abs(det.size_px - expected_size)

        # Allow a couple pixels of slack for thresholding/contour rounding.
        tol = 2.0
        if vert_err > tol or horiz_err > tol or size_err > tol:
            print(
                f"FAIL [{name}]: vertical_offset={det.vertical_offset_px:.1f} "
                f"(expected {expected_vert:.1f}, err {vert_err:.2f}), "
                f"horizontal_offset={det.horizontal_offset_px:.1f} "
                f"(expected {expected_horiz:.1f}, err {horiz_err:.2f}), "
                f"size={det.size_px:.1f} (expected {expected_size:.1f}, err {size_err:.2f})"
            )
            ok = False
        else:
            print(
                f"PASS [{name}]: vertical_offset={det.vertical_offset_px:+.1f}px "
                f"horizontal_offset={det.horizontal_offset_px:+.1f}px "
                f"size={det.size_px:.1f}px"
            )

    # Negative case: no marker in frame at all (uniform image).
    blank = np.full((height, width), 200, dtype=np.uint8)
    det = detect_marker(blank)
    if det.found:
        print("FAIL [blank frame]: falsely detected a marker where none exists")
        ok = False
    else:
        print("PASS [blank frame]: correctly reported not found")

    # Low-contrast case: marker present but barely distinguishable from bg.
    # Otsu should still split it, but this is the realistic-lighting probe.
    low_contrast = _make_synthetic_frame(width, height, 160, 120, 30, bg=180, fg=150)
    det = detect_marker(low_contrast)
    print(
        f"{'PASS' if det.found else 'INFO'} [low contrast, bg=180 fg=150]: "
        f"found={det.found}"
        + (f" size={det.size_px:.1f}px" if det.found else "")
    )

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", help="path to a grayscale frame")
    parser.add_argument("--self-test", action="store_true", help="run the synthetic self-test")
    args = parser.parse_args()

    if args.self_test or not args.image:
        ok = _self_test()
        print("\nSELF-TEST:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"Could not read image: {args.image}", file=sys.stderr)
        return 1

    det = detect_marker(gray)
    if not det.found:
        print("Marker not found")
        return 1

    print(f"found=True bbox={det.bbox}")
    print(f"center=({det.center_x:.1f}, {det.center_y:.1f})")
    print(f"vertical_offset_px={det.vertical_offset_px:+.2f}")
    print(f"horizontal_offset_px={det.horizontal_offset_px:+.2f}")
    print(f"size_px={det.size_px:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
