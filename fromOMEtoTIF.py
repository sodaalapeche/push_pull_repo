import tifffile
import os

# Input and output paths
input_path = "/home/chorus/Homogenous/push_pull-inj_3/push_pull-inj_3_MMStack_Defaultdebit08.ome.tif"  # replace with your input file
output_folder = "/home/chorus/expjuillet/homogeneppinj3/"  # replace with your output folder

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read the OME-TIFF
with tifffile.TiffFile(input_path) as tif:
    images = tif.asarray()  # loads all pages into a numpy array

# Check the shape (for debugging)
print(f"Image stack shape: {images.shape}, dtype: {images.dtype}")

# Save each frame as an individual TIFF, preserving 16-bit
for i, frame in enumerate(images):
    output_path = os.path.join(output_folder, f"frame_{i:04d}.tif")
    tifffile.imwrite(output_path, frame, dtype=frame.dtype)
    print(f"Saved {output_path}")
