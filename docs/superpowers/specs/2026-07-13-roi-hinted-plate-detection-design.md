# ROI-Hinted Plate Detection + Colab v0 — Notebook Spec

**Date:** 2026-07-13
**Status:** Approved design, pre-implementation
**Scope:** Polish `clonogenics.ipynb` into a usable notebook (ROI-hinted detection + bbox
drawing), then **deploy to Google Colab** as the bare-minimum v0 the lab can actually use.
**Parent:** `2026-07-13-clonogenics-gui-design.md` (the full GUI plan, pinned — v1 later).

## Why this exists

Two goals, one artifact — **"get the notebook well → deploy to Colab":**

1. **De-risk the core CV change.** The GUI design hinges on one unproven idea: replacing the
   brittle plate-cropping (`x_limit_frac` global cutoff + whole-image `detect_plate_rects`
   contour guessing) with **ROI-hinted scoped detection**. Prove it in the notebook on real
   scans before building an app around it.
2. **Ship a usable v0 now.** A polished notebook + bbox flow, run on **Colab (free GPU)** with
   the lab's **TIFFs on Google Drive**, is the absolute-minimum-effort way to make this usable
   for non-technical users — no packaging, no PyApp, no Tauri. The full app (parent spec) is
   the eventual upgrade; this gets them something working far sooner.

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

## Mechanism — race two approaches, keep the winner

This is a de-risk prototype, so **don't pick on theory** — build both A and B, run the shift
series through each, compare overlays against the gold `grid.png`, keep whichever wins.

Common to both, per image, per ROI hint:
1. **Pad the ROI** by a configurable margin (fraction of ROI size) to absorb hand-placement
   shift — the user draws on the true plate; the plate may shift by up to the margin in later
   scans.
2. Detect the precise plate box within the padded ROI (A or B, below); return `(x, y, w, h)`
   in full-image coords.
3. **Place wells** by rows×cols even-spaced fractions inside the detected box, then
   `refine_well` (unchanged) Hough-snaps each circle to its real ring.

### Approach A — scoped contour detection
Reuse the existing CLAHE → Canny → morph-close → contour machinery, masked/cropped to the
padded ROI, expecting **exactly one** plate: pick the dominant boxy contour (largest area
passing simple rectangularity/aspect checks, relative to the ROI). Reuses working code; but
inherits the faint-border / interior-clutter brittleness of the current whole-image detector.

### Approach B — edge-snap to the four sides (favored prior)
Exploits the accurate-draw prior: the true plate edges sit in narrow bands near each ROI side.
- Left/right edge: sum `|gx|` (vertical-edge strength) per column within the left/right search
  band → peak column = that edge. Top/bottom: sum `|gy|` per row in the top/bottom bands.
- Box = the 4 snapped edges.
- **Graceful degradation:** if a side has no strong edge peak, keep the drawn edge (already a
  good estimate since the user drew accurately) — A cannot do this.
- Directional aggregation amplifies a faint-but-consistent border over random interior specks,
  so it should tolerate faint transparent plate frames + colony/handwriting clutter better than
  contour-finding. Assumes near-axis-aligned plates (true for these scans).

### Shared fallback
If the chosen approach finds no usable box in an ROI, fall back to the padded ROI box itself,
so the pipeline always yields something and `refine_well` still corrects per-well.

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

## Delivery: Colab v0 (the "deploy" half)

Once the notebook runs well locally, deploy it as a Colab notebook the lab opens and runs.
Bare minimum — do not gold-plate:

- **GPU runtime** — notebook documents "Runtime → Change runtime type → GPU" (free T4).
- **Deps cell** — `pip install` torch / cellpose / ultralytics / jupyter-bbox-widget at the
  top (Colab re-runs this each session; ~minutes, unavoidable on ephemeral runtimes).
- **Google Drive for data** — `from google.colab import drive; drive.mount(...)`. Users drop
  their TIFF folder in Drive; the notebook points at that folder. Outputs (CSV + triptychs +
  zip) written **back to Drive** so nothing is lost when the runtime resets.
- **bbox ROI flow** — the `jupyter-bbox-widget` cell works in Colab; draw ROIs on the first
  scan, run the folder.
- **Detection knobs** — Cellpose params stay in a clearly-marked config cell they edit
  (FIJI-comfortable users are fine with this; the polished tune-loop UI is the parent app,
  not v0).
- **Simplicity for non-technical users** — hide code where practical, lead with a short
  "how to use" markdown cell (set GPU → mount Drive → point at folder → draw boxes → Run all
  → find results in Drive).

**Accepted v0 costs** (documented, not solved): Google account required; data lives on Google
Drive (leaves the lab); deps reinstall + no persistent env each session; notebook UX, not the
app GUI. These are the known trade-offs vs the full local app in the parent spec.

## Success criteria

**Validation data:**
- **Input scans:** `../Clonogenics-orig/*.tif` (~20 real ~100MB TIFFs across several experiments).
- **Gold reference:** `./batch_output_20260704_163617/<plate>/grid.png` — overlays from a full
  prior pipeline run, to compare new ROI-detection overlays against.
- **Primary shift test:** the within-experiment series `4h HS 43 + radiation` 001/002/004/005/006
  — same plate layout, real hand-placement shift between scans. Draw ROI hints once on 001,
  confirm they carry across the rest.

**CV (the de-risk):**
- Draw ROI hints once **on the true plates** (as accurately as the user would) on the first
  image of the series; the code pads them by the configurable margin to absorb shift.
- ROI-scoped detection lands correct plate boxes + wells on **every** image, visibly robust to
  shift, verified by the per-image grid overlay (lime box + red wells + labels).
- No regression vs the gold `grid.png` on the images the current pipeline already handled.

**Delivery (the v0):**
- The notebook runs end-to-end on Colab against a Drive folder, from mount → bbox → results
  back in Drive, driven by the how-to cell alone — no code editing beyond the marked config cell.

## Scope

**In:**
- `jupyter-bbox-widget` ROI capture cell → `roi_hints` (fractions) in the Config `profile`.
- **Both** detectors: `detect_plate_rect_in_roi_contour` (A) and `detect_plate_rect_in_roi_edges`
  (B), behind a switch, plus a comparison cell that renders A vs B overlays side by side against
  the gold `grid.png` on the shift series → pick the winner.
- Rewire `detect_wells` to loop over `roi_hints` (scoped detection) instead of
  `detect_plate_rects` + `x_limit_frac`.
- rows×cols → even-spaced fraction generation.
- Validation cell: run across a whole folder, render grid overlays.
- **Colab deployment:** deps cell, Drive mount, output-to-Drive, GPU + how-to markdown cells,
  clearly-marked config cell.

**Out:**
- Segmentation / Cellpose algorithm changes (v0 reuses the existing `count_colonies` as-is).
- App / API / frontend / the real GUI rect-drawing / the polished tune-loop UI.
- Removing the old `detect_plate_rects` (keep as a fallback path for now).
- Keeping *both* A and B long-term — the loser is dropped once the race is decided.
- Tests (the parent project defers a suite; revisit later).
- Solving the accepted v0 costs (Google account, data-on-Drive, per-session reinstall) — the
  full local app in the parent spec is where those go away.

## After this lands

- Ship the Colab notebook to the lab as v0.
- Reconcile the parent GUI spec's `detect.py` section with whatever the prototype actually
  proved, then unpin the GUI plan for the full local app.
