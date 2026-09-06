# AGENTS.md

Guidance for coding agents working in this repo.

## What this repo is

**`auto-clonogenics`** counts colonies (and measures average colony diameter) from multi-well
plate `.tif` scans. A single scan is a huge image (~100MB, roughly 5000×7000 px) holding several
plates, each a grid of wells (commonly 3×2). The pipeline turns each scan into per-well colony
counts.

The source scans live **outside** this repo, in the sibling directory `../Clonogenics` (some
docs reference `../Clonogenics-orig`). This repo contains code, model weights, and output
artifacts — not the input data.

## Current state (notebook reality)

The **active development surface is `clonogenics.ipynb`** — a Jupyter notebook that runs both
locally and on Google Colab. It is the *current* pipeline. Older standalone scripts
(`crop.py`, `ai_crop.py`) are legacy; the notebook has moved past them.

> Note: the existing `CLAUDE.md` was replaced by this file. It described the notebook as
> `Untitled.ipynb` and well-detection as FastSAM-based — both are stale. The notebook now uses
> **`wellcrop`** (with ROI hints) for well detection, not FastSAM.

## Pipeline

Two stages, both orchestrated inside `clonogenics.ipynb`:

### Stage 1 — Well isolation
- The user draws **one rectangle per plate** on a downscaled preview using
  `jupyter-bbox-widget`. These drawn boxes become resolution-independent **ROI hints**
  (fractions of image W/H), reused across every scan in a batch.
- `wellcrop.PlateDetector` takes the ROI hints and detects the precise plate box per image
  (scoped to each hint), places an even rows×cols grid of wells inside it, and Hough-snaps each
  circle to the real rim (`refine_well`). It returns `wells` (each with `.label`, `.x`, `.y`,
  `.radius`, `.mask`, `.image`) and `plate_boxes`.
- `well.extract_crop(img_rgb)` produces a **circular-masked RGB crop** per well (corners outside
  the rim are blacked out so Cellpose only sees the circle).

### Stage 2 — Colony segmentation (Cellpose)
- Builds a color-agnostic **LAB "signal" image**: distance of each pixel from the well-background
  (background sampled from the rim ring), normalized to 0–255.
- Zeroes out outlier pixels before segmentation: **debris** (darker than background by
  `L_DARK_MARGIN`) and **glint** (brighter than background by `L_BRIGHT_MARGIN`).
- Runs `models.CellposeModel(model_type='cpsam_v2').eval(...)` on the signal image.
- Outputs per-well `Colonies` count (plus a triptych PNG: raw well / outlier overlay / segmented).

## Running locally

- Package manager is **`uv`**. Python pinned `>=3.11,<3.12`.
- Install deps: `uv sync`
- Launch the notebook: `uv run jupyter lab` (or `uv run jupyter notebook`)
- Run scripts: `uv run python <script>.py <path_to_image.tif>`
- **Always use `uv run python ...`** (or `uv run jupyter ...`), never bare `python`/`python3`,
  so the venv is used.

### Local notebook config (not Colab)
In the `Dataset` cell the local (non-Colab) branch hardcodes:
- `INPUT_DIR = "../Clonogenics"` — edit to your local scans folder
- `REFERENCE_SCAN = "reference_scan.tif"` — edit to your reference scan filename
- `BATCH_SCANS = []` — `[]` runs every `.tif`/`.tiff` in `INPUT_DIR`; or list filenames for a subset

`FOLDER_URL` and `PLATE_ROWS`/`PLATE_COLS` must still be filled in the Config cell
(`PLATE_ROWS`/`PLATE_COLS` are validated and will raise if blank).

### Workflow
1. `uv sync` (once), then `uv run jupyter lab`.
2. In the Config cell, set `PLATE_ROWS`/`PLATE_COLS` (and `FOLDER_URL` only if running on Colab).
3. Run the notebook cells top to bottom.
4. In the `Dataset` cell, set `INPUT_DIR` and `REFERENCE_SCAN` for local scans.
5. Draw one box per plate on the bbox widget (Section 2).
6. **Runtime > Run all**, or run cell-by-cell.

There is **no test suite, no lint config, and no CI**.

## Two run modes

- **Tune (default)** — runs `REFERENCE_SCAN` alone with plots shown, so you can check counts
  before committing to a batch.
- **Batch** — sweeps every scan in `INPUT_DIR`, writes per-well PNGs + `batch_Results.csv` to
  `batch_output/`, and zips the output (`batch_output_<timestamp>.zip`).

## Tuning knobs (all in the Config cell)

These are the ones to adjust if results are wrong. **Well/plate detection** is separate from
**colony segmentation** — only touch detection constants if detection itself misbehaves.

### Well/plate detection
`MARGIN_FRAC`, `CLAHE_CLIP`, `REFINE_WELLS`, `REFINE_SEARCH_FRAC`, `REFINE_RADIUS_TOL`,
`REFINE_MAX_SHIFT`, `REFINE_DOWNSCALE_PX`, `HOUGH_PARAM1`, `HOUGH_PARAM2`,
`WELL_RADIUS_PITCH_FRAC`, `PLATE_SIZE_TOL`, `GRID_PITCH_TOL`, `MAX_GRID_ROTATION_DEGREES`,
`LOCK_TRUST_FRAC`.

### Cellpose colony segmentation
- `CELLPOSE_DIAMETER` (default 70) — expected colony diameter in px. Too large merges touching
  colonies; too small splits one colony into several.
- `CELLPOSE_FLOW_THRESHOLD` (0.9) — higher keeps more (rougher) masks.
- `CELLPOSE_CELLPROB_THRESHOLD` (-2.0) — lower (more negative) is more permissive.
- `CELLPOSE_NORM_LOW`/`CELLPOSE_NORM_HIGH` (1.0/99.0) — normalization percentile range.
- `CELLPOSE_MIN_COLONY_DIAMETER` (15) — pin-prick filter; converts to `CELLPOSE_MIN_SIZE`.

### LAB outlier (debris/glint) filtering — pre-Cellpose, not a Cellpose param
- `L_DARK_MARGIN` (97) — pixels darker than background by more than this are zeroed.
- `L_BRIGHT_MARGIN` (40) — pixels brighter than background by more than this are zeroed.

## Device / portability

The notebook picks the device at runtime (`mps` on Apple Silicon → `cuda` → `cpu`) — **never
hardcode `device='mps'` deep in logic**; keep it a parameter/config so the swap to a `cuda`/`cpu`
box is trivial. It will eventually run on a GTX 1650 Ti (4GB VRAM), so:
- Favor small/tiny model variants where possible.
- Avoid mps-only ops; prefer things that also run under cuda/cpu.
- Watch VRAM footprint (the model-loading cell skips reloading if already loaded).

## Dependencies

`cellpose`, `torch`, `ultralytics` (FastSAM), `mobile-sam`, `dinov3`, `wellcrop`, `opencv-python`,
`tifffile`, `numpy`, `pandas`, `jupyter`, `jupyter-bbox-widget`. Managed in `pyproject.toml` +
`uv.lock`.

## Repo layout (key files)

- `clonogenics.ipynb` — **the pipeline** (primary surface; Colab + local).
- `crop.py` — legacy dumb crop (left-60% slice). Superseded.
- `ai_crop.py` — legacy FastSAM-based well crop (`FastSAM-s.pt`). Superseded by `wellcrop`.
- `debug_plate_edges.py` — ad-hoc diagnostic (analytic circles vs Hough-refined circles).
- `FastSAM-s.pt`, `mobile_sam.pt` — model weights (gitignored via `*.pt`).
- `docs/superpowers/` — design specs + plans. The long-term direction is a local desktop app
  (FastAPI + React, packaged with PyApp) — the notebook is the de-risking prototype. These docs
  are partly historical/superseded by the current `wellcrop` approach.
- `batch_output*/`, `*.csv`, `*.zip`, `roi_preview.png` — gitignored output artifacts.

## Conventions

- No test suite / lint / CI yet.
- The notebook has a deliberate style: each function lives in its own cell, and expensive/stateful
  steps (model load, Drive listing, bbox widget) skip re-running when nothing changed.
- Output CSVs are named `*_Results.csv` for full runs and `*_wells.csv` for wells-only runs, so
  one doesn't overwrite the other.
