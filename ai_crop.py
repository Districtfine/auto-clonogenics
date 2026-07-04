
import sys
import os
import cv2
import numpy as np
import tifffile as tifi
from ultralytics import FastSAM

def main():
    if len(sys.argv) < 2:
        print("Usage: python ai_crop.py <path_to_image.tif>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        sys.exit(1)
        
    print(f"Reading {input_path}...")
    img = tifi.imread(input_path)
    
    # FastSAM AI expects standard RGB input
    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img[..., :3] 
        
    print("Loading FastSAM foundation model...")
    # This will automatically download the tiny 'FastSAM-s.pt' weights on the first run
    model = FastSAM('FastSAM-s.pt')
    
    print("Running neural network segmentation...")
    # imgsz=1024 safely downsamples the giant scan for the AI so it runs instantly on her M5
    # The AI automatically scales the resulting coordinates back up to her 7000x3000 resolution
    results = model.predict(source=img_rgb, device='mps', imgsz=1024, conf=0.15, iou=0.8, verbose=False)
    
    masks = results[0].masks
    if masks is None:
        print("Error: AI could not detect any distinct shapes.")
        sys.exit(1)
        
    img_area = img.shape[0] * img.shape[1]
    valid_wells = []
    
    # Evaluate every object the AI found
    for polygon in masks.xy:
        cnt = np.array(polygon, dtype=np.int32)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        if perimeter == 0: continue
        
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        # A well takes up roughly 1-6% of the total canvas and is highly circular
        if (img_area * 0.008) < area < (img_area * 0.08) and circularity > 0.70:
            # Get the exact center and radius of the AI's shape mask
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            
            # Prevent double-counting if the AI grabbed both the inner and outer plastic lip
            is_duplicate = False
            for (ex, ey, er) in valid_wells:
                dist = np.sqrt((cx - ex)**2 + (cy - ey)**2)
                if dist < r * 0.5:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                valid_wells.append((int(cx), int(cy), int(r)))
                
    print(f"AI detected {len(valid_wells)} distinct circular wells.")
    
    # Sort geometrically: grab the leftmost 12 (ignoring text/artifacts), then sort top-to-bottom
    valid_wells = sorted(valid_wells, key=lambda w: w[0])[:12]
    valid_wells = sorted(valid_wells, key=lambda w: (w[1] // 500, w[0]))
    
    labels = [
        "Top_A1", "Top_A2", "Top_B1", "Top_B2", "Top_C1", "Top_C2",
        "Bottom_A1", "Bottom_A2", "Bottom_B1", "Bottom_B2", "Bottom_C1", "Bottom_C2"
    ]
    
    dir_name, file_name = os.path.split(input_path)
    base_name = os.path.splitext(file_name)[0]
    
    for idx, (x, y, r) in enumerate(valid_wells):
        if idx >= len(labels): break
        label = labels[idx]
        
        # Shave 6% off the radius so the crop sits perfectly inside the plastic rim
        crop_r = int(r * 0.94)
        
        y_min, y_max = max(0, y - crop_r), min(img.shape[0], y + crop_r)
        x_min, x_max = max(0, x - crop_r), min(img.shape[1], x + crop_r)
        
        well_crop = img[y_min:y_max, x_min:x_max].copy()
        
        # Black out the corners so Cellpose only sees the pure circle
        mask = np.zeros(well_crop.shape[:2], dtype="uint8")
        cv2.circle(mask, (well_crop.shape[1] // 2, well_crop.shape[0] // 2), crop_r, 255, -1)
        final_well = cv2.bitwise_and(well_crop, well_crop, mask=mask)
        
        output_name = f"{base_name}_AI_Well_{label}.tif"
        tifi.imwrite(os.path.join(dir_name, output_name), final_well)
        print(f" Saved: {output_name}")
        
    print("Done! AI segmentation complete. 🧠🧫")

if __name__ == "__main__":
    main()
