"""
Blood smear cell counter
------------------------
Segments and counts cells in a microscopy image (e.g. a peripheral blood
smear) using classical image analysis: Otsu thresholding, morphological
cleanup and a distance-transform watershed to split touching cells.

Usage:
    python cell_counter.py --image path/to/smear.jpg --out results
    python cell_counter.py --demo --out results      # synthetic demo image

Outputs (in --out):
    overlay.png      original image with detected cell outlines + IDs
    area_hist.png    histogram of cell areas
    cells.csv        per-cell measurements (area, eccentricity, solidity)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import color, filters, io, measure, morphology, segmentation, feature


def make_demo_image(size=512, n_cells=45, seed=0):
    """Synthetic smear-like image: pink-ish background, darker round cells.
    Cells may touch but never overlap, so the true count is known."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    img = np.ones((size, size, 3)) * np.array([0.93, 0.80, 0.84])
    placed = []
    while len(placed) < n_cells:
        cx, cy = rng.integers(30, size - 30, 2)
        r = rng.integers(14, 22)
        if any((cx - x) ** 2 + (cy - y) ** 2 < (r + r2 + 1) ** 2 for x, y, r2 in placed):
            continue
        placed.append((cx, cy, r))
    for cx, cy, r in placed:
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        img[mask] = np.array([0.78, 0.30, 0.42])
        pallor = (xx - cx) ** 2 + (yy - cy) ** 2 <= (r * 0.4) ** 2  # central pallor
        img[pallor] = np.array([0.86, 0.50, 0.58])
    img += rng.normal(0, 0.015, img.shape)
    return np.clip(img, 0, 1), len(placed)


def segment_cells(img, min_area=80, min_distance=10):
    """Return a labeled mask where each cell has a unique integer id."""
    gray = color.rgb2gray(img)
    smooth = filters.gaussian(gray, sigma=1.5)
    thresh = filters.threshold_otsu(smooth)
    mask = smooth < thresh                       # cells are darker than background
    mask = ndi.binary_fill_holes(mask)           # fill central pallor
    mask = ndi.binary_opening(mask, structure=morphology.disk(2))
    lab = measure.label(mask)
    areas = np.bincount(lab.ravel())
    small = np.isin(lab, np.where(areas < min_area)[0])  # drop tiny debris
    mask = mask & ~small

    # Split touching cells with a distance-transform watershed
    dist = ndi.distance_transform_edt(mask)
    peaks = feature.peak_local_max(dist, min_distance=min_distance,
                                   labels=measure.label(mask))
    markers = np.zeros_like(mask, dtype=int)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = segmentation.watershed(-dist, markers, mask=mask)
    return labels


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="path to a microscopy image")
    ap.add_argument("--demo", action="store_true", help="use a synthetic image")
    ap.add_argument("--out", default="results", help="output folder")
    ap.add_argument("--min-area", type=int, default=80)
    args = ap.parse_args()

    true_count = None
    if args.demo:
        img, true_count = make_demo_image()
    elif args.image:
        img = io.imread(args.image)[..., :3]
        img = img / 255.0 if img.dtype == np.uint8 else img
    else:
        ap.error("provide --image PATH or --demo")

    os.makedirs(args.out, exist_ok=True)
    labels = segment_cells(img, min_area=args.min_area)
    props = measure.regionprops(labels)

    df = pd.DataFrame({
        "cell_id": [p.label for p in props],
        "area_px": [p.area for p in props],
        "eccentricity": [round(p.eccentricity, 3) for p in props],
        "solidity": [round(p.solidity, 3) for p in props],
    })
    df.to_csv(os.path.join(args.out, "cells.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(segmentation.mark_boundaries(img, labels, color=(0, 0.6, 0)))
    for p in props:
        ax.text(p.centroid[1], p.centroid[0], str(p.label), fontsize=6,
                ha="center", va="center", color="navy")
    ax.set_title(f"Detected cells: {len(props)}")
    ax.axis("off")
    fig.savefig(os.path.join(args.out, "overlay.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(df["area_px"], bins=20, color="steelblue", edgecolor="white")
    ax.set_xlabel("Cell area (pixels)")
    ax.set_ylabel("Count")
    ax.set_title("Cell size distribution")
    fig.savefig(os.path.join(args.out, "area_hist.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Cells detected: {len(props)}")
    if true_count is not None:
        print(f"Ground truth (synthetic demo): {true_count}")
    print(f"Mean area: {df['area_px'].mean():.1f} px  (SD {df['area_px'].std():.1f})")
    print(f"Results saved in: {args.out}/")


if __name__ == "__main__":
    main()
