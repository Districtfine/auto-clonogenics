"""Ad-hoc diagnostic: plate-boundary + analytic wells + local-Hough refinement on one scan.
Draws analytic circles (orange) vs refined circles (red) so we can see the snap. Run:
    uv run python debug_plate_edges.py
"""
import os
import cv2
import numpy as np
import tifffile as tifi
import matplotlib.pyplot as plt

INPUT_PATH = "../Clonogenics/4h HS 43 + radiation001.tif"
X_LIMIT_FRAC = 0.55
CLAHE_CLIP = 3.0
CANNY_LOW, CANNY_HIGH = 30, 90
PLATE_AREA_FRAC = (0.03, 0.45)
PLATE_ASPECT = (0.4, 2.2)
PLATE_RECTANGULARITY = 0.6
N_PLATES = 2
WELL_X_FRACS = [0.27, 0.73]
WELL_Y_FRACS = [0.19, 0.50, 0.81]
WELL_R_FRAC = 0.20
REFINE_SEARCH_FRAC = 0.5
REFINE_RADIUS_TOL = 0.2
REFINE_MAX_SHIFT = 0.5
HOUGH_PARAM1, HOUGH_PARAM2 = 100, 30
OUT_PATH = "batch_output/_debug_refine.png"


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img[..., :3]


def detect_plate_rects(gray, img_area):
    gray = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (img_area * PLATE_AREA_FRAC[0] <= area <= img_area * PLATE_AREA_FRAC[1]):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h if h else 0
        if not (PLATE_ASPECT[0] <= aspect <= PLATE_ASPECT[1]):
            continue
        if area / (w * h) < PLATE_RECTANGULARITY:
            continue
        candidates.append((area, x, y, w, h))
    candidates.sort(reverse=True)
    boxes = [(x, y, w, h) for _, x, y, w, h in candidates[:N_PLATES]]
    boxes.sort(key=lambda box: box[1] + box[3] / 2)
    return boxes


def refine_well(gray, cx, cy, r):
    pad = int(r * (1 + REFINE_SEARCH_FRAC))
    h, w = gray.shape
    x0, x1 = max(0, cx - pad), min(w, cx + pad)
    y0, y1 = max(0, cy - pad), min(h, cy + pad)
    roi = cv2.medianBlur(gray[y0:y1, x0:x1], 5)
    if roi.size == 0:
        return cx, cy, r, False
    circles = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(roi.shape),
                               param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
                               minRadius=int(r * (1 - REFINE_RADIUS_TOL)),
                               maxRadius=int(r * (1 + REFINE_RADIUS_TOL)))
    if circles is None:
        return cx, cy, r, False
    roi_cx, roi_cy = cx - x0, cy - y0
    fcx, fcy, fr = min(circles[0], key=lambda c: (c[0] - roi_cx) ** 2 + (c[1] - roi_cy) ** 2)
    if (fcx - roi_cx) ** 2 + (fcy - roi_cy) ** 2 > (r * REFINE_MAX_SHIFT) ** 2:
        return cx, cy, r, False
    return int(x0 + fcx), int(y0 + fcy), int(fr), True


def main():
    img_rgb = to_rgb(tifi.imread(INPUT_PATH))
    crop_x = int(img_rgb.shape[1] * X_LIMIT_FRAC)
    crop = img_rgb[:, :crop_x]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    refine_gray = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8)).apply(gray)
    boxes = detect_plate_rects(gray, gray.shape[0] * gray.shape[1])
    print(f"crop {gray.shape}, plates: {len(boxes)}")

    overlay = crop.copy()
    refined_count = 0
    for plate_idx, (px, py, pw, ph) in enumerate(boxes):
        cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (0, 255, 0), 6)
        well_r = int(pw * WELL_R_FRAC)
        for y_frac in WELL_Y_FRACS:
            for x_frac in WELL_X_FRACS:
                cx, cy = int(px + pw * x_frac), int(py + ph * y_frac)
                cv2.circle(overlay, (cx, cy), well_r, (255, 165, 0), 4)   # analytic = orange
                rcx, rcy, rr, ok = refine_well(refine_gray, cx, cy, well_r)
                refined_count += ok
                cv2.circle(overlay, (rcx, rcy), rr, (255, 0, 0), 5)       # refined = red
                cv2.circle(overlay, (rcx, rcy), 10, (255, 0, 0), -1)
    print(f"wells refined (snapped): {refined_count}/{len(boxes) * len(WELL_X_FRACS) * len(WELL_Y_FRACS)}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 12))
    ax.imshow(overlay)
    ax.set_title("orange = analytic, red = Hough-refined")
    ax.axis("off")
    fig.savefig(OUT_PATH, dpi=120, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
