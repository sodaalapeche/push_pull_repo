import tifffile
import numpy as np
import h5py
import skimage



#%%
ome_file=('/home/chorus/exp/heterogenous/ppull_sablefin_bille_1cm_09-12_3/ppull_sablefin_bille_1cm_09-12_3_MMStack_Default.ome.tif')
import skimage.io



def save_tiff_to_hdf5(tiff_path, hdf5_path, dataset_name="image"):
    # Read the OME-TIFF file
    image = skimage.io.imread(tiff_path)

    # Determine the appropriate dtype
    dtype = image.dtype

    # Save to HDF5
    with h5py.File(hdf5_path, "w") as hdf5_file:
        hdf5_file.create_dataset(dataset_name, data=image, dtype=dtype)

    print(f"Saved {tiff_path} to {hdf5_path} preserving dtype {dtype}")


#%%
save_tiff_to_hdf5(ome_file, '/home/chorus/exp/test.hdf5', "image")
