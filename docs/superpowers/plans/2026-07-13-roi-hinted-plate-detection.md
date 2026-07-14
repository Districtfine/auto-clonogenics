# ROI-Hinted Plate Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the brittle global-crop plate detection with ROI-hinted scoped detection, racing two approaches (contour vs edge-snap) in the notebook and keeping the winner.

**Architecture:** Extract the geometry pipeline (plate-box detection, well placement, ring refinement) out of `clonogenics.ipynb` into a pure, torch-free, importable module `roi_detect.py` (cv2 + numpy only). The user draws one rectangle per plate on the true plate; code pads it by a margin and re-detects the precise box per image (handling hand-placement shift) with either approach A (scoped contour) or approach B (edge-snap). Wells are placed by rows×cols even spacing inside the detected box, then `refine_well` Hough-snaps each to its ring. The notebook drives ROI capture (`jupyter-bbox-widget`) and an A-vs-B visual bake-off against the gold `grid.png`.

**Tech Stack:** Python 3.11, opencv-python, numpy, `jupyter-bbox-widget`, matplotlib, pytest (dev). Managed by `uv`.

## Global Constraints

- Package manager is `uv`; run everything via `uv run ...`, never bare `python`. (`uv run pytest ...`, `uv run jupyter lab`.)
- Python pinned `>=3.11,<3.12`.
- **No one-letter variable names** — descriptive names everywhere (loop targets, comprehensions, lambdas included). `i, j, k` allowed only for genuine integer indices; `x, y` only for spatial coordinates.
- **No ad-hoc inline interpreter scripts** (`python -c`, `uv run python -c`). Verify only by running the pytest suite or a notebook cell.
- Detection code is **torch-free** — cv2 + numpy only, no model loading, so tests run fast without a GPU.
- No globals in `roi_detect.py` — every tuning value is an explicit parameter with a default. Defaults copied verbatim from the notebook: `CLAHE_CLIP = 3.0`, `CANNY_LOW, CANNY_HIGH = 30, 90`, `PLATE_RECTANGULARITY = 0.6`, `HOUGH_PARAM1 = 100`, `HOUGH_PARAM2 = 30`, `REFINE_SEARCH_FRAC = 0.5`, `REFINE_RADIUS_TOL = 0.2`, `REFINE_MAX_SHIFT = 0.5`, `REFINE_DOWNSCALE_PX = 300`.
- Tests are deliberately minimal — deterministic unit tests for the pure geometry/detection math only. CV correctness on real scans is verified visually in the notebook against the gold `grid.png`, not by assertion.

## File Structure

- **Create `roi_detect.py`** (repo root, alongside existing `crop.py` / `ai_crop.py`) — pure cv2/numpy geometry: fraction↔pixel helpers, ROI padding, both detectors, dispatch, `refine_well` (moved out of the notebook), well placement, and the top-level `detect_wells_from_rois`.
- **Create `tests/test_roi_detect.py`** — deterministic pytest for every pure function, using synthetic images (numpy rectangles), no torch, no real TIFFs.
- **Modify `pyproject.toml`** — add `jupyter-bbox-widget` (main) and `pytest` (dev).
- **Modify `clonogenics.ipynb`** — add a config `roi_hints` block, a `jupyter-bbox-widget` capture cell, and an A-vs-B comparison/validation cell. The old `detect_plate_rects` / `detect_wells` stay as an untouched fallback path.

**Data:** input `../Clonogenics-orig/*.tif`; gold `./batch_output_20260704_163617/<plate>/grid.png`; primary shift series = `4h HS 43 + radiation` 001/002/004/005/006.

---

### Task 1: Module scaffolding + fraction/pixel/grid helpers

**Files:**
- Modify: `pyproject.toml`
- Create: `roi_detect.py`
- Test: `tests/test_roi_detect.py`

**Interfaces:**
- Produces:
  - `well_grid_fracs(rows: int, cols: int) -> tuple[list[float], list[float]]` — returns `(x_fracs, y_fracs)`, even-spaced cell centers.
  - `roi_frac_to_px(roi: dict, image_width: int, image_height: int) -> tuple[int,int,int,int]` — `(x0, y0, x1, y1)` from a `{"x","y","w","h"}` fractional box.
  - `pad_box(box_px: tuple[int,int,int,int], image_width: int, image_height: int, margin_frac: float) -> tuple[int,int,int,int]` — expands each side by `margin_frac` of the box's own width/height, clamped to the image.

- [ ] **Step 1: Add dependencies**

Edit `pyproject.toml`. In `[project].dependencies` add `"jupyter-bbox-widget>=0.5.0"`. Replace the empty dev list:

```toml
[tool.uv]
managed = true
dev-dependencies = ["pytest>=8.0"]
```

- [ ] **Step 2: Sync and confirm**

Run: `uv sync`
Expected: resolves and installs `jupyter-bbox-widget` and `pytest` with no error.

- [ ] **Step 3: Write the failing tests**

Create `tests/test_roi_detect.py`:

```python
import numpy as np
import pytest

import roi_detect


def test_well_grid_fracs_2x3_centers_are_evenly_spaced():
    x_fracs, y_fracs = roi_detect.well_grid_fracs(rows=3, cols=2)
    assert x_fracs == [0.25, 0.75]
    assert y_fracs == pytest.approx([1 / 6, 3 / 6, 5 / 6])


def test_well_grid_fracs_single_column():
    x_fracs, y_fracs = roi_detect.well_grid_fracs(rows=3, cols=1)
    assert x_fracs == [0.5]
    assert y_fracs == pytest.approx([1 / 6, 3 / 6, 5 / 6])


def test_roi_frac_to_px_scales_to_image():
    roi = {"x": 0.25, "y": 0.10, "w": 0.50, "h": 0.40}
    box = roi_detect.roi_frac_to_px(roi, image_width=1000, image_height=2000)
    assert box == (250, 200, 750, 1000)


def test_pad_box_expands_and_clamps_to_image():
    padded = roi_detect.pad_box((250, 200, 750, 1000), image_width=1000, image_height=2000, margin_frac=0.10)
    # width 500 -> 50 px each side; height 800 -> 80 px each side
    assert padded == (200, 120, 800, 1080)


def test_pad_box_clamps_at_edges():
    padded = roi_detect.pad_box((10, 10, 90, 90), image_width=100, image_height=100, margin_frac=0.50)
    assert padded == (0, 0, 100, 100)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_roi_detect.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'roi_detect'`.

- [ ] **Step 5: Implement the helpers**

Create `roi_detect.py`:

```python
"""ROI-hinted plate detection and well geometry — pure cv2/numpy, no torch.

The user draws one rectangle per plate on the true plate outline. Code pads that
rectangle by a margin (to absorb hand-placement shift between scans) and re-detects
the precise plate box inside it per image, then places wells by an even rows x cols
grid and snaps each to its ring. Two detectors are provided (contour, edge-snap);
see detect_plate_rect.
"""

import cv2
import numpy as np


def well_grid_fracs(rows, cols):
    """Even-spaced cell-center fractions inside a plate box: center of each grid cell."""
    x_fracs = [(col_index + 0.5) / cols for col_index in range(cols)]
    y_fracs = [(row_index + 0.5) / rows for row_index in range(rows)]
    return x_fracs, y_fracs


def roi_frac_to_px(roi, image_width, image_height):
    """Convert a fractional {x,y,w,h} box (each in [0,1]) to integer (x0,y0,x1,y1) pixels."""
    x0 = int(round(roi["x"] * image_width))
    y0 = int(round(roi["y"] * image_height))
    x1 = int(round((roi["x"] + roi["w"]) * image_width))
    y1 = int(round((roi["y"] + roi["h"]) * image_height))
    return x0, y0, x1, y1


def pad_box(box_px, image_width, image_height, margin_frac):
    """Expand a pixel box by margin_frac of its own width/height on each side, clamped."""
    x0, y0, x1, y1 = box_px
    pad_x = int(round((x1 - x0) * margin_frac))
    pad_y = int(round((y1 - y0) * margin_frac))
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(image_width, x1 + pad_x),
        min(image_height, y1 + pad_y),
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_roi_detect.py -v`
Expected: PASS (5 passed).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock roi_detect.py tests/test_roi_detect.py
git commit -m "Add roi_detect module scaffolding + fraction/pixel/grid helpers"
```

---

### Task 2: Approach B — edge-snap detector

**Files:**
- Modify: `roi_detect.py`
- Test: `tests/test_roi_detect.py`

**Interfaces:**
- Consumes: nothing from Task 1 at call time (operates on a grayscale image + boxes).
- Produces:
  - `detect_plate_rect_edges(gray, search_box, prior_box, band_frac=0.35, min_peak_ratio=2.0) -> tuple[int,int,int,int]` — returns the detected `(x, y, w, h)` in full-image pixels. `search_box` and `prior_box` are `(x0,y0,x1,y1)` absolute pixel boxes (padded search region, and the user's drawn box). For each of the four sides, finds the strongest edge line within a band near that side of `search_box`; if no side beats `min_peak_ratio × median` edge strength, that side falls back to the corresponding `prior_box` edge.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_roi_detect.py`:

```python
def _white_bordered_rect(image_height, image_width, top, left, bottom, right, thickness=3):
    """Black image with a white rectangle outline at the given pixel bounds."""
    image = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.rectangle(image, (left, top), (right, bottom), color=255, thickness=thickness)
    return image


def test_edge_snap_locks_onto_true_rectangle():
    image = _white_bordered_rect(600, 800, top=150, left=200, bottom=450, right=600)
    prior_box = (210, 160, 590, 440)          # drawn slightly inside the true rect
    search_box = (150, 100, 650, 500)         # padded search region
    x, y, width, height = roi_detect.detect_plate_rect_edges(image, search_box, prior_box)
    assert x == pytest.approx(200, abs=4)
    assert y == pytest.approx(150, abs=4)
    assert x + width == pytest.approx(600, abs=4)
    assert y + height == pytest.approx(450, abs=4)


def test_edge_snap_falls_back_to_prior_when_a_side_has_no_edge():
    # rectangle missing its bottom edge -> bottom must fall back to the prior box
    image = np.zeros((600, 800), dtype=np.uint8)
    cv2.line(image, (200, 150), (600, 150), 255, 3)   # top
    cv2.line(image, (200, 150), (200, 450), 255, 3)   # left
    cv2.line(image, (600, 150), (600, 450), 255, 3)   # right
    prior_box = (205, 155, 595, 430)
    search_box = (150, 100, 650, 500)
    x, y, width, height = roi_detect.detect_plate_rect_edges(image, search_box, prior_box)
    assert y + height == pytest.approx(430, abs=4)     # prior bottom, not a detected edge
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_roi_detect.py -k edge_snap -v`
Expected: FAIL — `AttributeError: module 'roi_detect' has no attribute 'detect_plate_rect_edges'`.

- [ ] **Step 3: Implement the edge-snap detector**

Append to `roi_detect.py`:

```python
def _snap_edge(strength_profile, band_length, from_start, prior_index, min_peak_ratio):
    """Pick the peak of an edge-strength profile within a band at one end, else the prior.

    strength_profile: 1D summed edge strength across the search crop (columns or rows).
    band_length: how many samples near the chosen end form the search band.
    from_start: True to search the leading band (left/top), False the trailing (right/bottom).
    prior_index: fallback position (in search-crop coordinates) if no strong edge is found.
    Returns an index in search-crop coordinates.
    """
    profile_median = float(np.median(strength_profile)) or 1.0
    if from_start:
        band = strength_profile[:band_length]
        peak_offset = int(np.argmax(band))
        peak_index = peak_offset
    else:
        band = strength_profile[len(strength_profile) - band_length:]
        peak_offset = int(np.argmax(band))
        peak_index = len(strength_profile) - band_length + peak_offset
    if band[peak_offset] < min_peak_ratio * profile_median:
        return prior_index
    return peak_index


def detect_plate_rect_edges(gray, search_box, prior_box, band_frac=0.35, min_peak_ratio=2.0):
    """Approach B: snap each of the four plate edges to the strongest line near that side.

    Aggregates edge strength ALONG each edge direction (sum |gradient| down columns for
    the vertical left/right edges, across rows for the horizontal top/bottom edges), so a
    faint-but-continuous plate wall beats scattered interior clutter. Any side without a
    clear peak falls back to the corresponding prior_box (drawn) edge.
    """
    search_x0, search_y0, search_x1, search_y1 = search_box
    prior_x0, prior_y0, prior_x1, prior_y1 = prior_box
    crop = gray[search_y0:search_y1, search_x0:search_x1]

    gradient_x = np.abs(cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=3))
    gradient_y = np.abs(cv2.Sobel(crop, cv2.CV_64F, 0, 1, ksize=3))
    column_strength = gradient_x.sum(axis=0)   # one value per column -> vertical edges
    row_strength = gradient_y.sum(axis=1)      # one value per row    -> horizontal edges

    crop_width = crop.shape[1]
    crop_height = crop.shape[0]
    band_width = max(1, int(crop_width * band_frac))
    band_height = max(1, int(crop_height * band_frac))

    left_local = _snap_edge(column_strength, band_width, True, prior_x0 - search_x0, min_peak_ratio)
    right_local = _snap_edge(column_strength, band_width, False, prior_x1 - search_x0, min_peak_ratio)
    top_local = _snap_edge(row_strength, band_height, True, prior_y0 - search_y0, min_peak_ratio)
    bottom_local = _snap_edge(row_strength, band_height, False, prior_y1 - search_y0, min_peak_ratio)

    left_x = search_x0 + left_local
    right_x = search_x0 + right_local
    top_y = search_y0 + top_local
    bottom_y = search_y0 + bottom_local
    return left_x, top_y, right_x - left_x, bottom_y - top_y
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_roi_detect.py -k edge_snap -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add roi_detect.py tests/test_roi_detect.py
git commit -m "Add approach B edge-snap plate detector"
```

---

### Task 3: Approach A — scoped contour detector

**Files:**
- Modify: `roi_detect.py`
- Test: `tests/test_roi_detect.py`

**Interfaces:**
- Produces:
  - `detect_plate_rect_contour(gray, search_box, clahe_clip=3.0, canny_low=30, canny_high=90, rectangularity_min=0.6, min_area_frac=0.05) -> tuple[int,int,int,int] | None` — CLAHE→Canny→close→contour scoped to `search_box`, returns the dominant boxy contour's bounding box in full-image pixels, or `None` if nothing qualifies.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_roi_detect.py`:

```python
def test_contour_detector_finds_filled_plate_rectangle():
    image = np.zeros((600, 800), dtype=np.uint8)
    cv2.rectangle(image, (200, 150), (600, 450), color=200, thickness=cv2.FILLED)
    search_box = (150, 100, 650, 500)
    box = roi_detect.detect_plate_rect_contour(image, search_box)
    assert box is not None
    x, y, width, height = box
    assert x == pytest.approx(200, abs=6)
    assert y == pytest.approx(150, abs=6)
    assert width == pytest.approx(400, abs=10)
    assert height == pytest.approx(300, abs=10)


def test_contour_detector_returns_none_on_blank_region():
    image = np.zeros((600, 800), dtype=np.uint8)
    search_box = (150, 100, 650, 500)
    assert roi_detect.detect_plate_rect_contour(image, search_box) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_roi_detect.py -k contour -v`
Expected: FAIL — `AttributeError: ... has no attribute 'detect_plate_rect_contour'`.

- [ ] **Step 3: Implement the contour detector**

Append to `roi_detect.py`:

```python
def detect_plate_rect_contour(gray, search_box, clahe_clip=3.0, canny_low=30,
                              canny_high=90, rectangularity_min=0.6, min_area_frac=0.05):
    """Approach A: reuse the CLAHE -> Canny -> close -> contour machinery, scoped to the ROI.

    Expects exactly one plate in search_box: returns the largest boxy contour's bounding
    box (in full-image pixels) whose fill ratio >= rectangularity_min and whose area is at
    least min_area_frac of the crop. Returns None if nothing qualifies.
    """
    search_x0, search_y0, search_x1, search_y1 = search_box
    crop = gray[search_y0:search_y1, search_x0:search_x1]
    crop_area = crop.shape[0] * crop.shape[1]

    contrast = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(crop)
    blurred = cv2.GaussianBlur(contrast, (7, 7), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_box = None
    best_area = 0.0
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        rect_x, rect_y, rect_width, rect_height = cv2.boundingRect(contour)
        bounding_area = rect_width * rect_height
        if bounding_area == 0:
            continue
        rectangularity = contour_area / bounding_area
        if rectangularity < rectangularity_min:
            continue
        if bounding_area < min_area_frac * crop_area:
            continue
        if bounding_area > best_area:
            best_area = bounding_area
            best_box = (search_x0 + rect_x, search_y0 + rect_y, rect_width, rect_height)
    return best_box
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_roi_detect.py -k contour -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add roi_detect.py tests/test_roi_detect.py
git commit -m "Add approach A scoped contour plate detector"
```

---

### Task 4: Dispatch, ring refinement, well placement, and full ROI pipeline

**Files:**
- Modify: `roi_detect.py`
- Test: `tests/test_roi_detect.py`

**Interfaces:**
- Consumes: `well_grid_fracs`, `roi_frac_to_px`, `pad_box`, `detect_plate_rect_edges`, `detect_plate_rect_contour` (Tasks 1–3).
- Produces:
  - `detect_plate_rect(gray, search_box, prior_box, method="edges", **params) -> tuple[int,int,int,int]` — dispatches to the edge or contour detector; on a `None`/failed contour result, falls back to `prior_box`. Always returns a box.
  - `refine_well(gray, center_x, center_y, radius, ...) -> tuple[int,int,int]` — the notebook's Hough ring-snap, moved verbatim (descriptive names), returns `(center_x, center_y, radius)`.
  - `place_wells(gray, refine_gray, plate_box, rows, cols, well_r_frac, plate_letter, refine=True) -> list[tuple[int,int,int,str]]` — even grid inside `plate_box`, each snapped by `refine_well` when `refine=True`. Returns `(x, y, r, label)` per well.
  - `detect_wells_from_rois(image_rgb, roi_hints, method="edges", margin_frac=0.10, well_r_frac=0.20, refine=True) -> list[tuple[int,int,int,str]]` — the top-level replacement for the notebook's `detect_plate_rects` + `x_limit_frac` path. `roi_hints` is a list of `{"x","y","w","h","rows","cols","letter"}` dicts (fractions + grid + label).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_roi_detect.py`:

```python
def test_detect_plate_rect_dispatch_contour_falls_back_to_prior():
    blank = np.zeros((600, 800), dtype=np.uint8)          # contour finds nothing
    prior_box = (205, 155, 595, 445)
    search_box = (150, 100, 650, 500)
    box = roi_detect.detect_plate_rect(blank, search_box, prior_box, method="contour")
    assert box == (205, 155, 595 - 205, 445 - 155)         # (x, y, w, h) from the prior


def test_place_wells_lays_out_even_grid_without_refine():
    blank = np.zeros((700, 900, 3), dtype=np.uint8)
    gray = cv2.cvtColor(blank, cv2.COLOR_RGB2GRAY)
    plate_box = (100, 100, 400, 600)                       # x, y, w, h
    wells = roi_detect.place_wells(gray, gray, plate_box, rows=3, cols=2,
                                   well_r_frac=0.20, plate_letter="A", refine=False)
    assert len(wells) == 6
    labels = [label for (_x, _y, _radius, label) in wells]
    assert labels == ["A1", "A2", "A3", "A4", "A5", "A6"]
    first_x, first_y, first_radius, _ = wells[0]
    assert first_x == 100 + int(400 * 0.25)                # col center 0.25
    assert first_y == 100 + int(600 * (1 / 6))             # row center 1/6
    assert first_radius == int(400 * 0.20)


def test_detect_wells_from_rois_counts_wells_across_two_plates():
    image_rgb = np.zeros((2000, 1500, 3), dtype=np.uint8)
    roi_hints = [
        {"x": 0.10, "y": 0.05, "w": 0.40, "h": 0.40, "rows": 3, "cols": 2, "letter": "A"},
        {"x": 0.10, "y": 0.50, "w": 0.40, "h": 0.40, "rows": 3, "cols": 2, "letter": "B"},
    ]
    wells = roi_detect.detect_wells_from_rois(image_rgb, roi_hints, method="edges", refine=False)
    assert len(wells) == 12
    assert {label for (_x, _y, _radius, label) in wells} == {
        "A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "B4", "B5", "B6",
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_roi_detect.py -k "dispatch or place_wells or from_rois" -v`
Expected: FAIL — missing `detect_plate_rect` / `place_wells` / `detect_wells_from_rois`.

- [ ] **Step 3: Implement dispatch, refine_well, placement, and the pipeline**

Append to `roi_detect.py` (the `refine_well` body is copied from `clonogenics.ipynb` with descriptive names; defaults match the notebook constants):

```python
def detect_plate_rect(gray, search_box, prior_box, method="edges", **params):
    """Dispatch to the chosen detector; always return a box (falling back to prior_box)."""
    if method == "edges":
        return detect_plate_rect_edges(gray, search_box, prior_box, **params)
    if method == "contour":
        detected = detect_plate_rect_contour(gray, search_box, **params)
        if detected is not None:
            return detected
        prior_x0, prior_y0, prior_x1, prior_y1 = prior_box
        return prior_x0, prior_y0, prior_x1 - prior_x0, prior_y1 - prior_y0
    raise ValueError(f"unknown detection method: {method!r}")


def refine_well(gray, center_x, center_y, radius, search_frac=0.5, radius_tol=0.2,
                max_shift=0.5, downscale_px=300, hough_param1=100, hough_param2=30):
    """Snap an analytic (center_x, center_y, radius) to the real well ring via local Hough."""
    pad = int(radius * (1 + search_frac))
    image_height, image_width = gray.shape
    x0, x1 = max(0, center_x - pad), min(image_width, center_x + pad)
    y0, y1 = max(0, center_y - pad), min(image_height, center_y + pad)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return center_x, center_y, radius

    roi = cv2.medianBlur(roi, 5)
    scale = min(1.0, downscale_px / max(roi.shape))
    search_roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else roi

    circles = cv2.HoughCircles(
        search_roi, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=max(search_roi.shape),
        param1=hough_param1, param2=hough_param2,
        minRadius=int(radius * (1 - radius_tol) * scale),
        maxRadius=int(radius * (1 + radius_tol) * scale),
    )
    if circles is None:
        return center_x, center_y, radius
    if scale < 1.0:
        circles = circles / scale

    roi_center_x, roi_center_y = center_x - x0, center_y - y0
    found_x, found_y, found_radius = min(
        circles[0], key=lambda circle: (circle[0] - roi_center_x) ** 2 + (circle[1] - roi_center_y) ** 2
    )
    if (found_x - roi_center_x) ** 2 + (found_y - roi_center_y) ** 2 > (radius * max_shift) ** 2:
        return center_x, center_y, radius
    return int(x0 + found_x), int(y0 + found_y), int(found_radius)


def place_wells(gray, refine_gray, plate_box, rows, cols, well_r_frac, plate_letter, refine=True):
    """Place an even rows x cols grid of wells inside plate_box, optionally ring-refined."""
    plate_x, plate_y, plate_width, plate_height = plate_box
    x_fracs, y_fracs = well_grid_fracs(rows, cols)
    base_radius = int(plate_width * well_r_frac)

    wells = []
    for row_index, y_frac in enumerate(y_fracs):
        for col_index, x_frac in enumerate(x_fracs):
            center_x = int(plate_x + plate_width * x_frac)
            center_y = int(plate_y + plate_height * y_frac)
            if refine:
                center_x, center_y, radius = refine_well(refine_gray, center_x, center_y, base_radius)
            else:
                radius = base_radius
            label = f"{plate_letter}{row_index * cols + col_index + 1}"
            wells.append((center_x, center_y, radius, label))
    return wells


def detect_wells_from_rois(image_rgb, roi_hints, method="edges", margin_frac=0.10,
                           well_r_frac=0.20, refine=True, clahe_clip=3.0):
    """ROI-hinted replacement for detect_plate_rects + x_limit_frac: detect a plate box per
    drawn ROI, place its wells, snap to rings. Returns (x, y, r, label) for every well."""
    image_height, image_width = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if image_rgb.ndim == 3 else image_rgb
    refine_gray = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(gray) if refine else gray

    all_wells = []
    for roi_index, roi in enumerate(roi_hints):
        prior_box = roi_frac_to_px(roi, image_width, image_height)
        search_box = pad_box(prior_box, image_width, image_height, margin_frac)
        plate_box = detect_plate_rect(gray, search_box, prior_box, method=method)   # (x, y, w, h)
        plate_letter = roi.get("letter") or chr(ord("A") + roi_index)
        all_wells.extend(
            place_wells(gray, refine_gray, plate_box, roi["rows"], roi["cols"],
                        well_r_frac, plate_letter, refine=refine)
        )
    return all_wells
```

`detect_plate_rect`, `detect_plate_rect_edges`, and `detect_plate_rect_contour` all return the
same `(x, y, w, h)` form, so `plate_box` feeds straight into `place_wells` with no conversion.

- [ ] **Step 4: Run the full test file**

Run: `uv run pytest tests/test_roi_detect.py -v`
Expected: PASS (all tests from Tasks 1–4 green).

- [ ] **Step 5: Commit**

```bash
git add roi_detect.py tests/test_roi_detect.py
git commit -m "Add detector dispatch, ring refinement, well placement, ROI pipeline"
```

---

### Task 5: Notebook — ROI config block + bbox capture cell

**Files:**
- Modify: `clonogenics.ipynb`

**Interfaces:**
- Consumes: `roi_detect.detect_wells_from_rois` (Task 4).
- Produces: a notebook variable `roi_hints` (list of `{"x","y","w","h","rows","cols","letter"}` dicts) for the validation cell.

This task has no pytest gate — its deliverable is a working capture cell verified by running it. Use `NotebookEdit` to insert cells.

- [ ] **Step 1: Add an ROI-config markdown + code cell** after the Config section

Markdown: "ROI hints — draw one box per plate on the true plate; code pads by `MARGIN_FRAC`."

Code cell:

```python
import roi_detect

# Validation dataset for the ROI-hinted detection prototype.
ROI_INPUT_DIR = "../Clonogenics-orig"
SHIFT_SERIES = [
    "4h HS 43 + radiation001.tif",
    "4h HS 43 + radiation002.tif",
    "4h HS 43 + radiation004.tif",
    "4h HS 43 + radiation005.tif",
    "4h HS 43 + radiation006.tif",
]
GOLD_DIR = "batch_output_20260704_163617"   # gold grid.png per plate for comparison

MARGIN_FRAC = 0.10       # ROI padding to absorb hand-placement shift
PLATE_ROWS, PLATE_COLS = 3, 2   # this dataset: two 3x2 plates
WELL_R_FRAC = 0.20
```

- [ ] **Step 2: Add the bbox capture cell**

```python
import os
import cv2
import numpy as np
import tifffile as tifi
from jupyter_bbox_widget import BBoxWidget

# Load the first scan, downscale to a displayable preview for the widget.
first_scan = tifi.imread(os.path.join(ROI_INPUT_DIR, SHIFT_SERIES[0]))
first_rgb = cv2.cvtColor(first_scan, cv2.COLOR_GRAY2RGB) if first_scan.ndim == 2 else first_scan[..., :3]
full_height, full_width = first_rgb.shape[:2]

preview_scale = 1200 / max(full_height, full_width)
preview = cv2.resize(first_rgb, None, fx=preview_scale, fy=preview_scale, interpolation=cv2.INTER_AREA)
preview_path = "roi_preview.png"
cv2.imwrite(preview_path, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

bbox_widget = BBoxWidget(image=preview_path, classes=["plate"])
bbox_widget
```

- [ ] **Step 3: Run and draw**

Run the cell, draw one rectangle on each plate of interest (top plate, then bottom), tight on the true plate outline.

- [ ] **Step 4: Add the cell that converts drawn boxes to `roi_hints`**

```python
# Convert preview-pixel boxes to image fractions, top-to-bottom, labeled A, B, ...
drawn = sorted(bbox_widget.bboxes, key=lambda box: box["y"])
roi_hints = []
for plate_index, box in enumerate(drawn):
    roi_hints.append({
        "x": box["x"] / preview.shape[1],
        "y": box["y"] / preview.shape[0],
        "w": box["width"] / preview.shape[1],
        "h": box["height"] / preview.shape[0],
        "rows": PLATE_ROWS,
        "cols": PLATE_COLS,
        "letter": chr(ord("A") + plate_index),
    })
print(f"Captured {len(roi_hints)} ROI hints:")
for roi in roi_hints:
    print(roi)
```

- [ ] **Step 5: Run and verify**

Run the cell.
Expected: prints one ROI hint per plate drawn, each with fractional `x,y,w,h` in `[0,1]` and `rows=3, cols=2`.

- [ ] **Step 6: Commit**

```bash
git add clonogenics.ipynb
git commit -m "Add notebook ROI config + bbox capture cell"
```

---

### Task 6: Notebook — A-vs-B bake-off against gold (the decision gate)

**Files:**
- Modify: `clonogenics.ipynb`

**Interfaces:**
- Consumes: `roi_hints` (Task 5), `roi_detect.detect_wells_from_rois` (Task 4).

No pytest gate — the deliverable is a visual comparison plus printed invariants that let a human pick the winner. This is the point of the prototype.

- [ ] **Step 1: Add a helper cell that renders one detection overlay**

```python
import matplotlib.pyplot as plt

def render_overlay(axis, image_rgb, roi_hints, wells, title):
    """Draw padded ROI (yellow), well circles (red) + labels over the scan on `axis`."""
    axis.imshow(image_rgb)
    axis.set_title(title)
    image_height, image_width = image_rgb.shape[:2]
    for roi in roi_hints:
        prior_box = roi_detect.roi_frac_to_px(roi, image_width, image_height)
        search_x0, search_y0, search_x1, search_y1 = roi_detect.pad_box(
            prior_box, image_width, image_height, MARGIN_FRAC)
        axis.add_patch(plt.Rectangle((search_x0, search_y0), search_x1 - search_x0,
                                     search_y1 - search_y0, fill=False, edgecolor="yellow", linewidth=1.5))
    for center_x, center_y, radius, label in wells:
        axis.add_patch(plt.Circle((center_x, center_y), radius, fill=False, edgecolor="red", linewidth=2))
        axis.text(center_x + 10, center_y + 10, label, color="yellow", fontsize=12, fontweight="bold")
    axis.set_aspect("equal")
```

- [ ] **Step 2: Add the bake-off cell**

```python
expected_well_count = sum(roi["rows"] * roi["cols"] for roi in roi_hints)

for scan_name in SHIFT_SERIES:
    scan = tifi.imread(os.path.join(ROI_INPUT_DIR, scan_name))
    scan_rgb = cv2.cvtColor(scan, cv2.COLOR_GRAY2RGB) if scan.ndim == 2 else scan[..., :3]

    wells_edges = roi_detect.detect_wells_from_rois(scan_rgb, roi_hints, method="edges",
                                                    margin_frac=MARGIN_FRAC, well_r_frac=WELL_R_FRAC)
    wells_contour = roi_detect.detect_wells_from_rois(scan_rgb, roi_hints, method="contour",
                                                      margin_frac=MARGIN_FRAC, well_r_frac=WELL_R_FRAC)

    figure, (axis_edges, axis_contour) = plt.subplots(1, 2, figsize=(20, 12))
    render_overlay(axis_edges, scan_rgb, roi_hints, wells_edges, f"{scan_name}  —  B: edge-snap")
    render_overlay(axis_contour, scan_rgb, roi_hints, wells_contour, f"{scan_name}  —  A: contour")
    plt.show()

    print(f"{scan_name}: edges wells={len(wells_edges)}, contour wells={len(wells_contour)}, "
          f"expected={expected_well_count}")
```

- [ ] **Step 3: Add a gold-reference cell**

```python
# Show the gold grid.png for the same plates alongside, to eyeball no-regression.
from PIL import Image

for scan_name in SHIFT_SERIES:
    gold_grid = os.path.join(GOLD_DIR, scan_name, "grid.png")
    if os.path.exists(gold_grid):
        figure, axis = plt.subplots(figsize=(10, 12))
        axis.imshow(Image.open(gold_grid))
        axis.set_title(f"GOLD: {scan_name}")
        axis.axis("off")
        plt.show()
    else:
        print(f"no gold grid for {scan_name}")
```

- [ ] **Step 4: Run the whole bake-off**

Run cells in order.
Expected: for each of the 5 scans, an edge-snap overlay next to a contour overlay, the printed well counts equal `expected_well_count` (12) for the method that placed all plates, and the gold `grid.png` for reference.

- [ ] **Step 5: Human decision checkpoint**

Eyeball, per method across all 5 shifted scans:
- Does the detected plate box hug the true plate (wells centered in each well) despite shift?
- Does it match the gold overlay's well positions?
- Which method (A or B) is more consistent across the series?

Record the winner in a final markdown cell (e.g. "Winner: edge-snap (B) — contour drifted on scans 004/005"). This decides which method the parent GUI app uses.

- [ ] **Step 6: Commit**

```bash
git add clonogenics.ipynb roi_preview.png
git commit -m "Add A-vs-B bake-off against gold grid overlays"
```

---

## Self-Review

**Spec coverage:**
- ROI-hinted scoped detection, padding, draw-on-true-plates → Tasks 1, 4 (`pad_box`, `detect_wells_from_rois`).
- Approach A (contour) → Task 3; Approach B (edge-snap, favored) → Task 2; race + compare vs gold → Task 6.
- Shared fallback to padded ROI box → Task 4 (`detect_plate_rect` contour fallback; edge-snap per-side prior fallback).
- rows×cols → even-spaced fractions → Task 1 (`well_grid_fracs`), used in Task 4.
- Two shift-correction layers (box detect + `refine_well`) → Task 4.
- `jupyter-bbox-widget` ROI source, downscale + fraction conversion → Task 5.
- Validation data paths + shift series + gold grid → Tasks 5, 6.
- Deploy / Colab → intentionally EXCLUDED per user instruction (parent spec's delivery half is not in this plan).
- Segmentation/Cellpose unchanged → not touched (detection-only).
- Old `detect_plate_rects` kept as fallback → not modified.

**Placeholder scan:** No TBD/TODO; every code and test step is complete.

**Type consistency:** Detectors (`detect_plate_rect_edges`, `detect_plate_rect_contour`, `detect_plate_rect`) all return `(x, y, w, h)`. `roi_frac_to_px` and `pad_box` use the two-corner `(x0,y0,x1,y1)` form; conversions between the two forms are explicit in `detect_wells_from_rois`. `roi_hints` dict shape (`x,y,w,h,rows,cols,letter`) is consistent across Tasks 4–6. `place_wells` returns `(x,y,r,label)`, matching the notebook's existing well tuple and the render helper's unpacking.
