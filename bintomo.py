import os
import cv2
import tifffile as tiff

input_folder = "/home/chorus/20260305_COLONNE_PETITJEAN_1cm_fine_00_Recon/20260305_COLONNE_PETITJEAN_1cm_fine_00_SlicesY/"
output_folder = "/home/chorus/20260305_COLONNE_PETITJEAN_1cm_fine_00_Recon/colonne_downscale/"
scale_factor = 4

os.makedirs(output_folder, exist_ok=True)

files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith((".tif", ".tiff"))])

for file in files:
    input_path = os.path.join(input_folder, file)
    output_path = os.path.join(output_folder, file)

    # read tiff
    img = tiff.imread(input_path)

    # compute new size
    new_width = img.shape[1] // scale_factor
    new_height = img.shape[0] // scale_factor

    # resize
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # save
    tiff.imwrite(output_path, resized)

    print(f"Processed {file}")

print("Done.")