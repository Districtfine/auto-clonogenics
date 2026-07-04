import sys
import os
import tifffile as tifi

def main():
    # Verify the user passed an image argument
    if len(sys.argv) < 2:
        print("Error: Please provide a file path.")
        print("Usage: python crop.py <path_to_image.tif>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        sys.exit(1)
        
    print(f"Reading {input_path}...")
    img = tifi.imread(input_path)
    
    # Slice the image, keeping 60% of the width to give the wells plenty of breathing room
    cut_point = int(img.shape[1] * 0.60)
    cropped = img[:, :cut_point]
    
    # Generate output filename (e.g., 'cropped_4h HS + 24h Chemo001.tif')
    dir_name, file_name = os.path.split(input_path)
    output_name = f"cropped_{file_name}"
    output_path = os.path.join(dir_name, output_name) if dir_name else output_name
    
    print(f"Saving cropped image to {output_path}...")
    tifi.imwrite(output_path, cropped)
    print("Done! 🎉 Restored to 60% clean vertical slice.")

if __name__ == "__main__":
    main()
