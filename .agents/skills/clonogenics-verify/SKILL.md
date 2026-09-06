---
name: clonogenics-verify
description: Verify clonogenics.ipynb pipeline changes against real scans from ../Clonogenics using wellcrop detection plus the real cpsam_v2 model in the eval kernel, with the exact ROI hints and pitfalls learned from real runs.
---

# Clonogenics real-image verification

How to verify changes to `clonogenics.ipynb` (colony-counting pipeline) against real data.
Synthetic images hide the failure modes that matter (opaque overlay layers painting black,
edge-well false positives, layout collisions at wrong figsize), so any behavioral notebook
change MUST be smoke-tested with this workflow before claiming it works.

## Ground rules

- The notebook JSON is the source of truth. Extract cell sources and `exec` them in the
  eval kernel with stubs for missing imports; do NOT reimplement pipeline logic in the test.
- Locate cells by content match, never by fixed index:
  ```python
  nb = json.load(open('clonogenics.ipynb'))
  src = next(''.join(c['source']) for c in nb['cells']
             if 'def count_colonies' in ''.join(c['source']) and c['cell_type'] == 'code')
  ```
- One-shot scripts outside the kernel: always `uv run python -` (venv pinned >=3.11,<3.12).

## Environment (eval kernel, `reset: true` for a fresh start)

```python
import matplotlib; matplotlib.use('Agg')
plt.rcParams['figure.figsize'] = [12, 6]   # matches the notebook Imports cell; without it,
                                           # panel titles collide at the 6.4" default
```

- Device: `mps` on the local Apple Silicon box; `cuda`/`cpu` elsewhere — never hardcode.
- Cellpose weights are cached at `~/.cellpose/models/cpsam_v2`; first load takes ~1 min.
  Reuse `cp_model` across eval cells; never reload within one session.
- Globals `count_colonies` needs: `np`, `cv2`, `ndimage` (scipy), `plt`, `cp_model`, `os`,
  `tqdm` (stub: iterator wrapper with `set_postfix`), `L_OUTLIER_MARGIN`, plus the
  `CELLPOSE_*` params — read their values from the notebook's Cellpose-parameters cell.

```python
class _TB:
    def __init__(self, it): self._it = it
    def __iter__(self): return iter(self._it)
    def set_postfix(self, **kw): pass
class FakeTqdm:
    def __call__(self, it, **kw): return _TB(it)
```

## Real-image pipeline

Scan + hints below are the known-good benchmark (user-drawn ROI hints, stable folder layout):

```python
SCAN = '/Users/arseniborisovs/Coding/Clonogenics/4h HS + 24h Chemo001.tif'
ROI_HINTS = [
    {'x': 0.11926605504587157, 'y': 0.09916666666666667, 'w': 0.39908256880733944,
     'h': 0.42333333333333334, 'rows': 3, 'cols': 2, 'letter': 'A'},
    {'x': 0.13302752293577982, 'y': 0.5475, 'w': 0.3944954128440367,
     'h': 0.4266666666666667, 'rows': 3, 'cols': 2, 'letter': 'B'},
]
```

```python
import tifffile as tifi, cv2
from wellcrop import PlateDetector
img = tifi.imread(SCAN)
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img[..., :3]
detector = PlateDetector(margin_frac=0.20, clahe_clip=3.0, refine_wells=True,
    radius_pitch_frac=0.45, plate_size_tol=0.15, grid_pitch_tol=0.25,
    max_grid_rotation_degrees=8.0, lock_trust_frac=0.25,
    hough_param1=100, hough_param2=30)
wells, plate_boxes = detector.detect(img_rgb, ROI_HINTS, return_boxes=True)
for w in wells:
    w.extract_crop(img_rgb)   # -> w.image (RGB crop), w.mask, w.label
```

Expect 12 wells (2 plates x 3x2). Reference per-well counts with default tuning
(cpsam_v2, mps): A2=116, A4=82, A6=106, B2=107, B4=103, B6=93; edge wells
A1=12, A5=10, B1=11, B5=7. Verify on a colony-dense well (A2); edge wells with
few colonies are the ones most prone to junk counts on tray plastic.

```python
from cellpose import models
import torch
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
cp_model = models.CellposeModel(gpu=device != 'cpu', device=torch.device(device), model_type='cpsam_v2')
```

Then `count_colonies(wells, name, show_plots=..., save_dir=tempfile.mkdtemp())`.

## Inspecting figure output

- Patch `plt.show = lambda **kw: None` to keep figures inspectable.
- Capture trap: with `show_plots=True` the function calls `plt.show()` and never
  `plt.close(fig)` — a patched `plt.close` capture list stays EMPTY. It only fills when
  `show_plots=False` (and `save_dir` set, which also forces plotting). Use `plt.gcf()`
  for the show path.
- Panel 3 (AI segmentation) contract:
  - `len(fig.axes[2].images) == 3` — raw crop, mask fill (`alpha=0.3`), outlined copy.
  - `np.asarray(fig.axes[2].images[0].get_array())` must be pixel-identical to `well.image`
    (raw crop at the bottom; catches any opaque layer regression painting it black).
  - `fig.axes[2].texts` == 0 when `SHOW_COLONY_LABELS` is False (default) or
    == colony count when True.
- Outline-density thresholds must scale with colony count: a 116-colony well legitimately
  has ~7% outline pixels. Do not assert `< 0.05` on dense wells (calibrate ~`< 0.15`).
- Display figures inline for visual confirmation; eval auto-displays open figures.

## Notebook editing pitfalls

- The user's live Jupyter session overwrites the file on save; edits can vanish mid-task.
  Warn the user to File > Reload Notebook from Disk before continuing after file edits.
- The `edit` tool binds to a content hash and rejects when those concurrent writes churn
  the file. When it rejects repeatedly, fall back to scripted JSON edits: assert the exact
  anchor string exists, replace, write, re-load + `compile(dedented_source)` to verify.
- Cell `source` must stay a list of lines, each ending `\n` (`splitlines(keepends=True)`).
- `json.dump(nb, f, indent=1, ensure_ascii=False)` then a trailing `\n`.
- Config/param values must be added to the earlier config cell (runs before
  `count_colonies`); verify definition-before-use by comparing matched cell indices.
