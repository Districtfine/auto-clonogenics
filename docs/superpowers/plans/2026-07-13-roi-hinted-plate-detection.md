# ROI-Hinted Plate Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the brittle global-crop plate detection with ROI-hinted scoped detection, racing two approaches (contour vs edge-snap) entirely inside `clonogenics.ipynb`, and keep the winner.

**Architecture:** All new code lives in `clonogenics.ipynb` as new cells (matching the existing notebook style — functions defined in cells). The user draws one rectangle per plate on the true plate outline with `jupyter-bbox-widget`; code pads it by a margin and re-detects the precise plate box per image (handling hand-placement shift) with either approach A (scoped contour) or approach B (edge-snap). Wells are placed by an even rows×cols grid inside the detected box, then the notebook's **existing** `refine_well` Hough-snaps each to its ring. A final cell renders an A-vs-B bake-off against the gold `grid.png`.

**Tech Stack:** Python 3.11, opencv-python, numpy, matplotlib, tifffile, `jupyter-bbox-widget`. Managed by `uv`. No test suite, no new module — notebook only.

## Global Constraints

- Package manager is `uv`; run via `uv run ...`. Launch the notebook with `uv run jupyter lab`.
- Python pinned `>=3.11,<3.12`.
- **No one-letter variable names** — descriptive names everywhere (loop targets, comprehensions, lambdas included). `i, j, k` only for genuine integer indices; `x, y` only for spatial coordinates.
- **No ad-hoc inline interpreter scripts** (`python -c`, `uv run python -c`). Verification is running the notebook cells themselves — the notebook is the entry point.
- **No tests, no separate module.** Every function is a notebook cell.
- Detection code is torch-free (cv2 + numpy). It runs in the notebook's `WELLS_ONLY`-style path, so no GPU/Cellpose needed to validate.
- New cells are inserted **after** the notebook's existing `refine_well` / `detect_wells` definitions (Section 3), so `refine_well` and the config constants (`REFINE_*`, `HOUGH_PARAM1/2`, `CLAHE_CLIP`) are already in scope and reused, not redefined.

## What gets added to the notebook

All in `clonogenics.ipynb`, after Section 3 (well detection):
- A geometry-helpers cell: `well_grid_fracs`, `roi_frac_to_px`, `pad_box`.
- An edge-snap detector cell: `snap_edge`, `detect_plate_rect_edges` (approach B).
- A contour detector cell: `detect_plate_rect_contour` (approach A).
- A pipeline cell: `detect_plate_rect` (dispatch) + `place_wells` + `detect_wells_from_rois`, reusing the existing `refine_well`.
- An ROI-config cell + a `jupyter-bbox-widget` capture cell → `roi_hints`.
- An A-vs-B bake-off cell rendering overlays against the gold `grid.png` → pick the winner.

The existing `detect_plate_rects` / `detect_wells` / `x_limit_frac` path stays untouched as a fallback.

**Data:** input `../Clonogenics-orig/*.tif`; gold `./batch_output_20260704_163617/<plate>/grid.png`; primary shift series = `4h HS 43 + radiation` 001/002/004/005/006.

Use `NotebookEdit` to insert every cell. Each task ends by running the new cell(s) with no error and committing.

---

### Task 1: Add the bbox-widget dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the dependency**

Edit `pyproject.toml`, in `[project].dependencies` add:

```toml
    "jupyter-bbox-widget>=0.5.0",
```

- [ ] **Step 2: Sync**

Run: `uv sync`
Expected: resolves and installs `jupyter-bbox-widget` (and `anywidget`/`ipywidgets`) with no error.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add jupyter-bbox-widget dependency"
```

---

### Task 2: Geometry helpers cell

**Files:**
- Modify: `clonogenics.ipynb`

**Interfaces:**
- Produces (notebook scope): `well_grid_fracs(rows, cols) -> (x_fracs, y_fracs)`, `roi_frac_to_px(roi, image_width, image_height) -> (x0,y0,x1,y1)`, `pad_box(box_px, image_width, image_height, margin_frac) -> (x0,y0,x1,y1)`.

- [ ] **Step 1: Insert a markdown cell** after Section 3

Text: "## 3b. ROI-hinted plate detection (prototype) — geometry helpers. Draw one box per plate on the true plate outline; code pads by a margin and re-detects the precise box per image."

- [ ] **Step 2: Insert the helpers code cell**

```python
def well_grid_fracs(rows, cols):
    """Even-spaced cell-center fractions inside a plate box (center of each grid cell)."""
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

- [ ] **Step 3: Run the cell**

Expected: defines the three functions, no error, no output.

- [ ] **Step 4: Commit**

```bash
git add clonogenics.ipynb
git commit -m "Add ROI geometry helper cell to notebook"
```

---

### Task 3: Edge-snap detector cell (approach B)

**Files:**
- Modify: `clonogenics.ipynb`

**Interfaces:**
- Produces: `detect_plate_rect_edges(gray, search_box, prior_box, band_frac=0.35, min_peak_ratio=2.0) -> (x, y, w, h)`. `search_box`/`prior_box` are `(x0,y0,x1,y1)` absolute pixel boxes (padded search region; user's drawn box). Each side snaps to the strongest edge line in a band near that side of `search_box`; a side with no peak ≥ `min_peak_ratio × median` falls back to the corresponding `prior_box` edge.

- [ ] **Step 1: Insert the edge-snap code cell** (after Task 2's cell)

```python
def snap_edge(strength_profile, band_length, from_start, prior_index, min_peak_ratio):
    """Pick the peak of an edge-strength profile within a band at one end, else the prior.

    strength_profile: 1D summed edge strength across the search crop (columns or rows).
    band_length: number of samples near the chosen end that form the search band.
    from_start: True searches the leading band (left/top), False the trailing (right/bottom).
    prior_index: fallback position (in search-crop coords) when no strong edge is found.
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

    Sums |gradient| ALONG each edge direction (down columns for the vertical left/right
    edges, across rows for the horizontal top/bottom edges), so a faint-but-continuous
    plate wall beats scattered interior clutter. Any side without a clear peak falls back
    to the corresponding prior_box (drawn) edge. Returns (x, y, w, h) in full-image pixels.
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

    left_local = snap_edge(column_strength, band_width, True, prior_x0 - search_x0, min_peak_ratio)
    right_local = snap_edge(column_strength, band_width, False, prior_x1 - search_x0, min_peak_ratio)
    top_local = snap_edge(row_strength, band_height, True, prior_y0 - search_y0, min_peak_ratio)
    bottom_local = snap_edge(row_strength, band_height, False, prior_y1 - search_y0, min_peak_ratio)

    left_x = search_x0 + left_local
    right_x = search_x0 + right_local
    top_y = search_y0 + top_local
    bottom_y = search_y0 + bottom_local
    return left_x, top_y, right_x - left_x, bottom_y - top_y
```

- [ ] **Step 2: Run the cell**

Expected: defines `snap_edge` and `detect_plate_rect_edges`, no error.

- [ ] **Step 3: Commit**

```bash
git add clonogenics.ipynb
git commit -m "Add approach B edge-snap detector cell"
```

---

### Task 4: Scoped contour detector cell (approach A)

**Files:**
- Modify: `clonogenics.ipynb`

**Interfaces:**
- Produces: `detect_plate_rect_contour(gray, search_box, clahe_clip=CLAHE_CLIP, canny_low=CANNY_LOW, canny_high=CANNY_HIGH, rectangularity_min=PLATE_RECTANGULARITY, min_area_frac=0.05) -> (x, y, w, h) | None`. Reuses the notebook's existing config constants as defaults.

- [ ] **Step 1: Insert the contour code cell** (after Task 3's cell)

```python
def detect_plate_rect_contour(gray, search_box, clahe_clip=CLAHE_CLIP, canny_low=CANNY_LOW,
                              canny_high=CANNY_HIGH, rectangularity_min=PLATE_RECTANGULARITY,
                              min_area_frac=0.05):
    """Approach A: reuse the CLAHE -> Canny -> close -> contour machinery, scoped to the ROI.

    Expects exactly one plate in search_box: returns the largest boxy contour's bounding box
    (in full-image pixels) whose fill ratio >= rectangularity_min and whose area is at least
    min_area_frac of the crop. Returns None if nothing qualifies.
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

- [ ] **Step 2: Run the cell**

Expected: defines `detect_plate_rect_contour`, no error.

- [ ] **Step 3: Commit**

```bash
git add clonogenics.ipynb
git commit -m "Add approach A scoped contour detector cell"
```

---

### Task 5: Dispatch + well placement + ROI pipeline cell

**Files:**
- Modify: `clonogenics.ipynb`

**Interfaces:**
- Consumes: `well_grid_fracs`, `roi_frac_to_px`, `pad_box`, `detect_plate_rect_edges`, `detect_plate_rect_contour` (Tasks 2–4), and the **existing** notebook `refine_well`.
- Produces:
  - `detect_plate_rect(gray, search_box, prior_box, method="edges", **params) -> (x, y, w, h)` — dispatch; on a `None` contour result, falls back to `prior_box`.
  - `place_wells(gray, refine_gray, plate_box, rows, cols, well_r_frac, plate_letter, refine=True) -> list[(x, y, r, label)]`.
  - `detect_wells_from_rois(image_rgb, roi_hints, method="edges", margin_frac=0.10, well_r_frac=0.20, refine=True) -> list[(x, y, r, label)]`. `roi_hints` = list of `{"x","y","w","h","rows","cols","letter"}`.

- [ ] **Step 1: Insert the pipeline code cell** (after Task 4's cell)

```python
def detect_plate_rect(gray, search_box, prior_box, method="edges", **params):
    """Dispatch to the chosen detector; always return an (x, y, w, h) box."""
    if method == "edges":
        return detect_plate_rect_edges(gray, search_box, prior_box, **params)
    if method == "contour":
        detected = detect_plate_rect_contour(gray, search_box, **params)
        if detected is not None:
            return detected
        prior_x0, prior_y0, prior_x1, prior_y1 = prior_box
        return prior_x0, prior_y0, prior_x1 - prior_x0, prior_y1 - prior_y0
    raise ValueError(f"unknown detection method: {method!r}")


def place_wells(gray, refine_gray, plate_box, rows, cols, well_r_frac, plate_letter, refine=True):
    """Even rows x cols grid of wells inside plate_box, each snapped by the existing refine_well."""
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
                           well_r_frac=0.20, refine=True):
    """ROI-hinted replacement for detect_plate_rects + x_limit_frac: detect a plate box per
    drawn ROI, place its wells, snap to rings. Returns (x, y, r, label) for every well."""
    image_height, image_width = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if image_rgb.ndim == 3 else image_rgb
    refine_gray = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8)).apply(gray) if refine else gray

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

- [ ] **Step 2: Run the cell**

Expected: defines `detect_plate_rect`, `place_wells`, `detect_wells_from_rois`, no error. (Requires the existing `refine_well` cell to have been run earlier in the session.)

- [ ] **Step 3: Commit**

```bash
git add clonogenics.ipynb
git commit -m "Add ROI detection dispatch + well placement + pipeline cell"
```

---

### Task 6: ROI config + bbox capture cells

**Files:**
- Modify: `clonogenics.ipynb`

**Interfaces:**
- Consumes: nothing from earlier tasks at definition time.
- Produces (notebook scope): `ROI_INPUT_DIR`, `SHIFT_SERIES`, `GOLD_DIR`, `MARGIN_FRAC`, `PLATE_ROWS`, `PLATE_COLS`, `WELL_R_FRAC`, and `roi_hints` (list of `{"x","y","w","h","rows","cols","letter"}`).

- [ ] **Step 1: Insert the ROI-config code cell**

```python
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

MARGIN_FRAC = 0.10              # ROI padding to absorb hand-placement shift
PLATE_ROWS, PLATE_COLS = 3, 2   # this dataset: two 3x2 plates
WELL_R_FRAC = 0.20
```

- [ ] **Step 2: Insert the bbox capture cell**

```python
from jupyter_bbox_widget import BBoxWidget

# Load the first scan, downscale to a displayable preview for the widget.
first_scan = tifi.imread(os.path.join(ROI_INPUT_DIR, SHIFT_SERIES[0]))
first_rgb = cv2.cvtColor(first_scan, cv2.COLOR_GRAY2RGB) if first_scan.ndim == 2 else first_scan[..., :3]

preview_scale = 1200 / max(first_rgb.shape[0], first_rgb.shape[1])
preview = cv2.resize(first_rgb, None, fx=preview_scale, fy=preview_scale, interpolation=cv2.INTER_AREA)
preview_path = "roi_preview.png"
cv2.imwrite(preview_path, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

bbox_widget = BBoxWidget(image=preview_path, classes=["plate"])
bbox_widget
```

- [ ] **Step 3: Run and draw**

Run the cell; draw one rectangle on each plate of interest (top plate, then bottom), tight on the true plate outline.

- [ ] **Step 4: Insert the cell that builds `roi_hints`**

```python
# Convert preview-pixel boxes to image fractions, top-to-bottom, labeled A, B, ...
drawn_boxes = sorted(bbox_widget.bboxes, key=lambda box: box["y"])
roi_hints = []
for plate_index, box in enumerate(drawn_boxes):
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

Expected: prints one ROI hint per plate drawn, each with fractional `x,y,w,h` in `[0,1]` and `rows=3, cols=2`.

- [ ] **Step 6: Commit**

```bash
git add clonogenics.ipynb roi_preview.png
git commit -m "Add ROI config + bbox capture cells"
```

---

### Task 7: A-vs-B bake-off against gold (the decision gate)

**Files:**
- Modify: `clonogenics.ipynb`

**Interfaces:**
- Consumes: `roi_hints` (Task 6), `detect_wells_from_rois`, `roi_frac_to_px`, `pad_box` (Tasks 2, 5).

- [ ] **Step 1: Insert the overlay helper cell**

```python
def render_overlay(axis, image_rgb, roi_hints, wells, title):
    """Draw padded ROI (yellow), well circles (red) + labels over the scan on `axis`."""
    axis.imshow(image_rgb)
    axis.set_title(title)
    image_height, image_width = image_rgb.shape[:2]
    for roi in roi_hints:
        prior_box = roi_frac_to_px(roi, image_width, image_height)
        search_x0, search_y0, search_x1, search_y1 = pad_box(prior_box, image_width, image_height, MARGIN_FRAC)
        axis.add_patch(plt.Rectangle((search_x0, search_y0), search_x1 - search_x0,
                                     search_y1 - search_y0, fill=False, edgecolor="yellow", linewidth=1.5))
    for center_x, center_y, radius, label in wells:
        axis.add_patch(plt.Circle((center_x, center_y), radius, fill=False, edgecolor="red", linewidth=2))
        axis.text(center_x + 10, center_y + 10, label, color="yellow", fontsize=12, fontweight="bold")
    axis.set_aspect("equal")
```

- [ ] **Step 2: Insert the bake-off cell**

```python
expected_well_count = sum(roi["rows"] * roi["cols"] for roi in roi_hints)

for scan_name in SHIFT_SERIES:
    scan = tifi.imread(os.path.join(ROI_INPUT_DIR, scan_name))
    scan_rgb = cv2.cvtColor(scan, cv2.COLOR_GRAY2RGB) if scan.ndim == 2 else scan[..., :3]

    wells_edges = detect_wells_from_rois(scan_rgb, roi_hints, method="edges",
                                         margin_frac=MARGIN_FRAC, well_r_frac=WELL_R_FRAC)
    wells_contour = detect_wells_from_rois(scan_rgb, roi_hints, method="contour",
                                           margin_frac=MARGIN_FRAC, well_r_frac=WELL_R_FRAC)

    figure, (axis_edges, axis_contour) = plt.subplots(1, 2, figsize=(20, 12))
    render_overlay(axis_edges, scan_rgb, roi_hints, wells_edges, f"{scan_name}  -  B: edge-snap")
    render_overlay(axis_contour, scan_rgb, roi_hints, wells_contour, f"{scan_name}  -  A: contour")
    plt.show()

    print(f"{scan_name}: edges wells={len(wells_edges)}, contour wells={len(wells_contour)}, "
          f"expected={expected_well_count}")
```

- [ ] **Step 3: Insert the gold-reference cell**

```python
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

Run the three cells in order.
Expected: for each of the 5 scans, an edge-snap overlay beside a contour overlay; printed well counts equal `expected_well_count` (12) for whichever method placed all plates; then the gold `grid.png` per scan.

- [ ] **Step 5: Human decision checkpoint**

Eyeball, per method across all 5 shifted scans:
- Does the detected plate box hug the true plate (wells centered in each well) despite shift?
- Does it match the gold overlay's well positions?
- Which method (A or B) is more consistent across the series?

Record the winner in a final markdown cell (e.g. "Winner: edge-snap (B) — contour drifted on scans 004/005"). This decides which method the parent GUI app uses.

- [ ] **Step 6: Commit**

```bash
git add clonogenics.ipynb
git commit -m "Add A-vs-B bake-off against gold grid overlays"
```

---

## Self-Review

**Spec coverage:**
- ROI-hinted scoped detection + padding + draw-on-true-plates → Tasks 2, 5 (`pad_box`, `detect_wells_from_rois`).
- Approach A (contour) → Task 4; Approach B (edge-snap, favored) → Task 3; race + compare vs gold → Task 7.
- Shared fallback to prior/ROI box → Task 5 (contour `None` → prior; edge-snap per-side prior fallback in Task 3).
- rows×cols → even-spaced fractions → Task 2 (`well_grid_fracs`), used in Task 5.
- Two shift-correction layers (box detect + existing `refine_well`) → Task 5.
- `jupyter-bbox-widget` ROI source + downscale + fraction conversion → Task 6.
- Validation data paths + shift series + gold grid → Tasks 6, 7.
- No module, no tests → honored (all notebook cells; verification is running them).
- Deploy / Colab → EXCLUDED per user instruction.
- Segmentation/Cellpose + old `detect_plate_rects` → untouched (detection-only, fallback preserved).

**Placeholder scan:** No TBD/TODO; every cell is complete code.

**Type consistency:** `detect_plate_rect_edges`, `detect_plate_rect_contour`, `detect_plate_rect` all return `(x, y, w, h)`; `roi_frac_to_px`/`pad_box` use the `(x0,y0,x1,y1)` form; the one conversion (prior box → w,h) is explicit in `detect_plate_rect`'s contour fallback. `roi_hints` dict shape is consistent across Tasks 5–7. `place_wells` returns `(x, y, r, label)`, matching the notebook's existing well tuple and `render_overlay`'s unpacking. `refine_well` is the notebook's existing function, reused unchanged.
