# Clonogenics GUI — Design Spec

**Date:** 2026-07-13
**Status:** Approved design, pre-implementation
**Author:** the project owner (with Claude)

## Goal

Turn the `clonogenics.ipynb` proof-of-concept into a desktop app that non-technical
molecular-biology lab users (Windows + Mac) can install and run without a CLI, Python,
or dependency management. The app counts colonies and measures average colony diameter
from multi-well plate `.tif` scans, reproducing the current notebook workflow through a GUI.

### Users & constraints

- Users are lab scientists, comfortable with FIJI-style tools (they want real knobs, not
  dumbed-down sliders). Not comfortable with a terminal or Python environments.
- Fully **local** compute — no server, no cloud, no budget, no data-sharing agreements.
  Some machines have an NVIDIA GPU (e.g. GTX 1650 Ti, 4GB), some are CPU/MPS-only.
- Volunteer project, take-it-or-leave-it. Favor robustness and low maintenance over polish.

## Decisions locked in brainstorming

| Area | Decision | Why |
|---|---|---|
| Compute | Fully local, per-machine | No budget/server; data stays in lab |
| GUI tech | Local web app: **FastAPI backend + React (Vite/TS) frontend**, **shadcn/ui** (Radix + Tailwind) for components | Author knows this stack; canvas drawing needs HTML `<canvas>`; shadcn = copy-in components, no runtime lock-in |
| Packaging | **PyApp** binary (per-OS) | Real double-click binary, NOT a shell script; torch downloaded at runtime by uv, NO PyInstaller bundling hell; auto-picks correct torch wheel (CUDA/MPS/CPU) per machine |
| Desktop shell | **Deferred to phase-2 (Tauri)** | Ship the GUI first; Tauri layers on later with the PyApp binary as its sidecar — nothing wasted |
| Folder pick | **tkinter native dialog** (`askdirectory`) | ~10 lines, returns a real path (no uploading 100MB TIFFs); tk ships in python-build-standalone; swap for Tauri dialog in phase-2 |

### Why PyApp, not PyInstaller or a shell script

Two separate problems were conflated:
- **"Packaging hell"** = freezing torch/CUDA into a binary (PyInstaller's disease).
  Fixed by NOT bundling — deps download at runtime via uv.
- **"Rando shell script"** = a `.command`/`.bat` looks sketchy / trust problem.
  Fixed by shipping a real executable.

PyApp gets both: a real native binary (~few MB) that on first run downloads a standalone
Python + `uv`-installs the app and its deps into a user cache; later runs are instant.
- Private app shipped by embedding a local wheel (`PYAPP_PROJECT_PATH`).
- GPU per machine via `PYAPP_PIP_EXTRA_ARGS=--extra-index-url <pytorch-url>`.
- Signing wall (Gatekeeper/SmartScreen) is universal to unsigned apps and unavoidable
  without paid certs; mitigated by a one-time right-click→Open (Mac) / "Run anyway" (Win),
  documented with a short GIF for users.

## Architecture

```
v1 (this spec):
┌─ PyApp binary (per-OS) ────────────────────────────┐
│  entrypoint: start uvicorn → open browser tab       │
│                                                     │
│  FastAPI (127.0.0.1:PORT)                           │
│   ├─ serves built React bundle (static)             │
│   └─ JSON API                                       │
│        └─ pipeline (torch: FastSAM + Cellpose)      │
│                                                     │
│  React (Vite/TS) in a normal browser tab            │
└─────────────────────────────────────────────────────┘

phase-2 (later): Tauri shell wraps this — PyApp binary becomes Tauri's
sidecar, Tauri serves React + provides native window/dialogs/installer.
```

### Repo shape

```
auto-clonogenics/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI: serves React bundle + API; opens browser
│   │   ├── pipeline/
│   │   │   ├── models.py      # lazy-load FastSAM + Cellpose ONCE at startup
│   │   │   ├── detect.py      # user plate rects + grid dims → well centers → refine_well snap
│   │   │   └── segment.py     # LAB-distance signal + Cellpose → count + avg diameter
│   │   ├── profiles.py        # sqlite CRUD (SQLModel) + import/export
│   │   ├── jobs.py            # batch run, progress, cancel
│   │   ├── logging.py         # global exception capture → log file
│   │   └── api.py             # HTTP routes
│   └── pyproject.toml
├── frontend/                  # React + Vite + TS + Tailwind + shadcn/ui
│   └── src/…                  # components/ui (shadcn), screens, canvas
├── FastSAM-s.pt               # model weight, shipped in the wheel
└── packaging/                 # PyApp build config + GitHub Actions matrix (mac + win)
```

- **Notebook → library.** Pipeline code moves OUT of `clonogenics.ipynb` into
  `backend/app/pipeline/` as plain functions. The notebook stays as a dev scratchpad;
  the app never imports it.
- **Models load once** at startup (torch init is slow), held in memory, reused.
- **Boot handshake:** first launch, PyApp downloads Python+torch (GBs) before FastAPI is up.
  React polls `/health` and shows a "First-time setup, downloading models… (one time)"
  screen until the backend answers. Normal launches clear the poll in seconds.
- **Portable device pick** stays as in the notebook: `mps` → `cuda` → `cpu`, as config,
  never hardcoded deep in logic.

## Pipeline (extracted from the notebook)

The notebook already implements everything in three modes; the GUI replaces editing the
Config cell with API calls:

| Notebook mode | GUI screen |
|---|---|
| `WELLS_ONLY=True` (detect wells, check geometry) | Tune screen — geometry section |
| `BATCH_MODE=False` (single file, plots on) | Tune screen — detection section |
| `BATCH_MODE=True` (sweep folder) | Run screen |

- `detect.py` ← `refine_well` + well-grid generation. **Change:** plate geometry comes
  from **user-drawn rectangles + grid dims (rows×cols)**, not the hand-tuned `profile`.
  `detect_plate_rects` auto-detect is kept as an optional "auto-place rects" seed button.
  `refine_well` (Hough snap to the real rim) is kept as-is.
- `segment.py` ← LAB-distance signal + `count_colonies`. Returns count **and avg diameter**
  per well (measured from the Cellpose masks already produced).
- `models.py` ← load FastSAM + Cellpose once.
- **Live preview** = the existing `show_plots=True` single-plate path, wrapped as an
  endpoint that returns count(s) + the triptych PNG(s). Two granularities: **one selected
  well** (snappy) and **all wells** (full check), user-toggled.
- **Raw params exposed** (grouped as in the Config cell; notebook docstrings become
  tooltips):
  - Cellpose: `diameter`, `flow_threshold`, `cellprob_threshold`, `normalize_percentile[2]`
  - LAB outlier: `dark_margin`, `bright_margin`
  - Well refine/Hough: `refine_wells`, `refine_search_frac`, `refine_radius_tol`,
    `refine_max_shift`, `refine_downscale_px`, `hough_param1`, `hough_param2`

## Screens

Three screens: **Home**, **Tune**, **Run**.

### 1. Home / folder pick
- "Choose scan folder" → tkinter dialog → shows path + list of `.tif`s found.
- Select a saved profile (dropdown) or "New profile". Import/export profile buttons here too.
- No `.tif` in folder → clear message.

### 2. Tune (the core screen — geometry + detection stacked, never swap screens)

```
[◀ img: "24h 1um chemo001.tif" ▼]   profile: NT-6w   [Save ✓]
── GEOMETRY ──────────────────────────────────────────
  canvas: selected image           plate 1: rows[3] cols[2]
  draw/drag plate rects            plate 2: rows[3] cols[2]
  red well circles overlay live    well radius frac [..]
                                   [+ add plate rect] [auto-place]
── DETECTION ─────────────────────────────────────────
  preview: ( selected well | all wells )   ← toggle
  raw/outlier/AI-count triptych    | params: diameter, flow_thr,
  (click a well above to select)   |         cellprob, dark/bright,
                                   |         hough…
```

- **Workflow:** draw plate rects + grid dims → well circles snap → click a well → triptych +
  count → turn raw knobs → preview re-runs → **switch image** to verify the config carries →
  nudge if needed → repeat until no tuning needed. All on ONE screen.
- **Image switch = auto-run the saved config on the new image** and show the result. The
  whole point is verifying that one image's config generalizes; users only touch anything
  if it doesn't carry over.
- **Geometry across images:** plate rects stored as **fractions of image W/H**
  (resolution-independent), reused as the default on every image. Detection params carry
  over untouched. Nudging rects on an offset image updates the shared profile geometry.
- **Save** persists geometry + params to the profile.

### 3. Run
- "Run all N scans" → per-plate progress bar + live log, cancelable (stop button).
- On finish: results table (Plate / Well / Colonies / AvgDiameter), "Open output folder".

## Data model (SQLModel + sqlite)

One `profiles` table, stable columns + JSON blobs (so adding a knob needs no migration):

```
profiles
  id            int pk
  name          text
  source_folder text                # last folder used
  created_at / updated_at
  geometry      json  # plate_rects [{x,y,w,h as W/H fractions}], per-plate {rows,cols},
                      #   well_radius_frac, labeling scheme
  params        json  # cellpose {diameter, flow_threshold, cellprob_threshold,
                      #   normalize_percentile}, lab {dark_margin, bright_margin},
                      #   refine/hough {refine_*, hough_param1, hough_param2}
```

- DB lives in the OS app-data dir (`platformdirs`), survives app updates.
- **Import/export:** a profile serializes to/from a `.json` file so the lab can share tunes
  by email. Import validates shape, inserts as a new profile.

## Output

Reuses the current `batch_output/` layout so nothing downstream changes:

```
<output>/<plate_name>/A1.png … grid.png   # triptychs + grid overlay
<output>/batch_Results.csv                 # Plate, Well, Colonies, AvgDiameter (diameter added)
<output>_<timestamp>.zip                    # self-contained, for sending
```

- Output dir defaults next to the input folder. "Open output folder" button on finish.

## Errors & logging (minimal — volunteer project)

- **Global exception handler** → append to a rotating log file in the app-data dir.
- **"Export logs" button** → zips logs + last traceback for the user to send to the author.
- Cheap soft-fails kept as low priority (do not gold-plate):
  - one well/plate errors in Cellpose → mark that well `errored` in CSV, continue the batch;
  - CUDA OOM on a small-VRAM box → catch, fall back to CPU for that run, warn.
- Backend-not-ready is covered by the `/health` poll + first-run screen.

## Out of scope (v1)

- Tauri desktop shell (phase-2; layers onto PyApp cleanly).
- Native folder dialog via Tauri (v1 uses tkinter).
- Automated test suite (deferred; a pipeline golden-count test + profile CRUD test are the
  natural first additions when revisited).
- Built-in plots/stats — users do stats in Prism/Excel from the CSV.
- Code signing / notarization (needs paid certs; users do a one-time trust click).

## Phasing

1. **Backend pipeline extraction** — notebook → `backend/app/pipeline/` modules, driven by
   explicit geometry + params args instead of the global `profile`.
2. **FastAPI API** — profiles CRUD + import/export, folder pick, live preview endpoint,
   batch job with progress/cancel, health, logs export.
3. **React frontend** — Home, Tune (geometry + detection), Run.
4. **PyApp packaging** — wheel build, PyApp config (uv + torch extra-index), GitHub Actions
   matrix (mac + win), user-facing "how to open an unsigned app" GIF.
5. **(phase-2)** Tauri shell + native dialog; test suite.
