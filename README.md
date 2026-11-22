# Blood Smear Cell Counter

A small Python tool that segments and counts cells in microscopy images (such as peripheral blood smears) using classical image analysis. I built it to bridge my background in medical laboratory science (hematology, blood film preparation and interpretation) with biomedical image analysis.

## What it does
1. Converts the image to grayscale and smooths it (Gaussian filter)
2. Separates cells from background with Otsu thresholding
3. Cleans the mask (fills central pallor, removes debris)
4. Splits touching cells with a distance-transform watershed
5. Measures each cell (area, eccentricity, solidity) and saves results

## Quick start
```bash
pip install -r requirements.txt

# Run on the built-in synthetic demo image
python cell_counter.py --demo --out examples

# Run on your own image
python cell_counter.py --image my_smear.jpg --out results
```

## Outputs
- `overlay.png` – image with detected cell outlines and IDs
- `area_hist.png` – cell size distribution
- `cells.csv` – per-cell measurements

## Demo result (synthetic image)
The `--demo` image contains 45 generated cells with known ground truth. The tool detects 45/45. This only checks that the pipeline works; the demo is **not** a real blood smear and the result does not show performance on real data.

![overlay](examples/overlay.png)

## Limitations
- Thresholds are tuned for stained smears with darker cells on a lighter background; other stains or lighting may need parameter changes (`--min-area`).
- Heavily overlapping cells can be under-counted.
- Classical methods only; no deep learning.

## Planned next steps
- Evaluate on a public dataset (e.g. BCCD or Raabin-WBC) against annotated counts
- Add white blood cell vs red blood cell separation
- Compare with a pretrained deep-learning segmentation model

## Author
Ahmad Ibrahim, Medical Laboratory Science graduate, University of Jordan
