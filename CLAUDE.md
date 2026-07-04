# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Dev environment

Developing on Apple Silicon Mac (mps device). Will eventually move to wife's PC with a GTX 1650 Ti (cuda, 4GB VRAM). No need to change anything now, but keep model/approach choices portable:
- Don't hardcode `device='mps'` deep in logic — keep it a parameter/config so `cuda`/`cpu` swap is trivial.
- Avoid mps-only ops; prefer things that also run under cuda/cpu.
- Watch VRAM footprint (1650 Ti has only 4GB) when picking model sizes — favor small/tiny variants (already doing this with FastSAM-s, mobile_sam).

## Goal

Count colonies and average colony diameter from multi-well plate `.tiff` scans in `../Clonogenics` (sibling dir, not in this repo). Pipeline: isolate each well from the full plate scan, then run each well crop through Cellpose for colony segmentation/counting. Primary development surface is a Jupyter notebook (`Untitled.ipynb` — still unnamed/WIP).

## Setup / commands

- Package manager: `uv` (pyproject.toml + uv.lock). Python pinned `>=3.11,<3.12`.
- Install: `uv sync`
- Run notebook: `uv run jupyter lab` (or `jupyter notebook`)
- Run scripts: `uv run python <script>.py <path_to_image.tif>`
- Always use `uv run python ...`, not bare `python3`/`python` — keeps deps consistent with the venv.
- No test suite, lint config, or CI yet.

## Architecture / pipeline

Two-stage well isolation feeding into Cellpose segmentation:

1. **Well cropping** — isolate individual wells from a full plate `.tif` scan (source images live in `../Clonogenics`, large ~100MB multi-well TIFFs).
   - `crop.py`: dumb crop, slices off the left 60% of image width as a fixed heuristic (no model).
   - `ai_crop.py`: model-based crop using **FastSAM** (`FastSAM-s.pt`, ultralytics). Detects well shapes by circularity/area filtering on segmentation polygons, dedupes overlapping detections, geometrically sorts into a fixed 12-well layout (`Top_A1`...`Bottom_C2`), masks each crop to a circle (blacks out corners outside the well rim), writes one `<basename>_AI_Well_<label>.tif` per well.
   - `mobile_sam.pt` present as an alternative/lighter SAM backbone — not yet wired into a script.
2. **Colony segmentation** — cropped well TIFFs get run through **Cellpose** (dependency in pyproject.toml) to segment/count colonies and measure diameter. This stage currently lives in the notebook, not yet extracted to a script.

Model weight files (`FastSAM-s.pt`, `mobile_sam.pt`) are checked into the repo root directly rather than downloaded on demand.
