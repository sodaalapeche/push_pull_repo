import os
import tifffile
import numpy as np
from pathlib import Path
import h5py
from sympy.codegen.ast import uint32
from tqdm import tqdm
import imageio
import cv2
import skimage



def extract_images_from_ome_tif(root_folder):
    root_path = Path(root_folder)

    # Recherche récursive des fichiers .ome.tif
    ome_files = list(root_path.rglob("*.ome.tif"))

    if not ome_files:
        print("Aucun fichier .ome.tif trouvé.")
        return

    for ome_file in tqdm(ome_files, desc="Traitement des fichiers"):
        output_folder = ome_file.parent / "extracted_images"
        output_folder.mkdir(exist_ok=True)  # Crée le dossier s'il n'existe pas
        output_folder=str(output_folder)
        # Read the OME-TIFF file
        image = skimage.io.imread(ome_file)

        # Determine the appropriate dtype
        dtype = image.dtype

        # Save to HDF5
        with h5py.File(output_folder+'.hdf5', "w") as hdf5_file:
            hdf5_file.create_dataset(output_folder+'.hdf5', data=image, dtype=dtype,chunks=(10,image.shape[1],image.shape[2]), compression="gzip")

        print(f"Saved {output_folder+'.hdf5'} to {output_folder} preserving dtype {dtype}")

if __name__ == "__main__":
    extract_images_from_ome_tif("/home/chorus/exp/")

    
