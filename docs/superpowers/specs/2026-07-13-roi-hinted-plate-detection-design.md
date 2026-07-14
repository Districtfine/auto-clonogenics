# ROI-Hinted Plate Detection — Notebook Prototype Spec

**Date:** 2026-07-13
**Status:** Approved design, pre-implementation
**Scope:** Notebook prototype only. De-risks the core CV change before any app work.
**Parent:** `2026-07-13-clonogenics-gui-design.md` (the GUI plan, pinned until this lands).

## Why this exists

The GUI design hinges on one unproven CV idea: replacing the current brittle plate-cropping
(`x_limit_frac` global cutoff + whole-image `detect_plate_rects` contour guessing) with
**ROI-hinted scoped detection**. Rather than build a whole app around an unproven mechanism,
prove it in `clonogenics.ipynb` on real scans first. If it doesn't generalize across a folder
with real hand-placement shift, better to learn that in a notebook cell.

## The idea

The user (eventually, in the GUI) draws an approximate rectangle around each plate of interest.
That rectangle is a **region-of-interest hint, NOT the final plate box.** Plates are placed
into the scanner by hand → they shift between scans and are never pixel-identical → per-image
auto-detection must stay. Each drawn rect just tells detection "a plate lives roughly *here*;
hone in within this box, ignore everything else."

This replaces:
- `x_limit_frac` (the brittle global crop that excludes the right-side handwriting plates), and
- the whole-image, multi-plate 3-tier fallback with area/aspect/rectangularity filters relative
  to the full image.

## Mechanism (approach A — scoped contour detection)

Per image, per ROI hint:

1. **Pad the ROI** by a configurable margin (fraction of ROI size) to absorb hand-placement
   shift — the true plate may poke slightly outside where the user drew.
2. **Scoped detection** — reuse the existing CLAHE → Canny → morphological-close → contour
   machinery, but **masked/cropped to the padded ROI**, expecting **exactly one** plate: pick
   the dominant boxy contour (largest area passing simple rectangularity/aspect sanity checks,
   relative to the ROI not the whole image). Return the precise `(x, y, w, h)` in full-image
   coords.
   - If no usable contour is found in the ROI, fall back to the padded ROI box itself (so the
     pipeline always yields something, and `refine_well` still corrects per-well).
3. **Place wells** by rows×cols even-spaced fractions inside the detected box, then
   `refine_well` (unchanged) Hough-snaps each circle to its real ring.

**Approach B (edge-snap to the ROI's four sides)** is a documented fallback if a dataset fights
approach A. Not built in this prototype unless A proves insufficient.

### Two cleanly separated shift-correction layers (keep both)

- **ROI-scoped detection** re-anchors the *box* per image → handles gross hand-placement shift.
- **`refine_well`** locks each *well* to its ring → handles fine shift.

### Grid dims → fractions

rows×cols generates even-spaced well centers inside the detected box:
`x_fracs = [(col + 0.5) / cols for col in range(cols)]`, same for rows. Replaces the
hand-authored `well_x_fracs` / `well_y_fracs`. `well_r_frac` stays a knob; `refine_well`
corrects imperfect placement.

## ROI-hint source in notebook-land: `jupyter-bbox-widget`

No GUI yet, so use [`jupyter-bbox-widget`](https://pypi.org/project/jupyter-bbox-widget/) as
the stand-in for the future drawing UI — a truer test than hand-authoring coords.

```python
from jupyter_bbox_widget import BBoxWidget
widget = BBoxWidget(image='preview.png', classes=['plate'])
# draw boxes; then:
widget.bboxes  # → [{'x','y','width','height','label'}, ...]
```

- Draw one box per plate of interest on the **first scan**.
- **Downscale** the ~5000×7000 TIFF to a displayable PNG preview for the widget; convert the
  returned pixel coords to **fractions of image W/H** (resolution-independent) → store as
  `profile['roi_hints']`.
- Dep: `anywidget` / `ipywidgets` (light). Added to `pyproject.toml`.

## Success criteria

On **one real folder of scans with visible plate shift between images** (from `../Clonogenics`):
- Draw ROI hints once (deliberately looser than the true plates) on the first image.
- ROI-scoped detection lands correct plate boxes + wells on **every** image in the folder,
  visibly robust to shift, verified by the per-image grid overlay (lime box + red wells + labels).
- No regression vs the current output on the images it already handled.

## Scope

**In:**
- `jupyter-bbox-widget` ROI capture cell → `roi_hints` (fractions) in the Config `profile`.
- `detect_plate_rect_in_roi(img_rgb, roi, profile)` — approach A.
- Rewire `detect_wells` to loop over `roi_hints` (scoped detection) instead of
  `detect_plate_rects` + `x_limit_frac`.
- rows×cols → even-spaced fraction generation.
- Validation cell: run across a whole folder, render grid overlays.

**Out:**
- Segmentation / Cellpose changes.
- App / API / frontend / the real GUI rect-drawing.
- Removing the old `detect_plate_rects` (keep as a fallback path for now).
- Approach B edge-snap (only if A proves insufficient).
- Tests (the parent project defers a suite; revisit later).

## After this lands

Reconcile the parent GUI spec's `detect.py` section with whatever the prototype actually
proved, then unpin the GUI plan.
