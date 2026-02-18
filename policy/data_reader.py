import h5py
import numpy as np
import os

# Replace with the path to your HDF5 file
file_path = '/home/haozhe/Dropbox/imitationlearning/act_demo_data/sim_transfer_cube_scripted-20250402T233905Z-002/sim_transfer_cube_scripted/episode_16.hdf5'

output_dir = 'hdf5_txt_output'
os.makedirs(output_dir, exist_ok=True)

with h5py.File(file_path, 'r') as f:
    def recursively_save(name, obj):
        if isinstance(obj, h5py.Dataset):
            data = obj[:]
            txt_filename = os.path.join(output_dir, f"{name.replace('/', '_')}.txt")
            
            with open(txt_filename, 'w') as txt_file:
                txt_file.write(f"# Dataset: {name}\n")
                txt_file.write(f"# Shape: {data.shape}\n")
                txt_file.write(f"# Dtype: {data.dtype}\n\n")

                if data.ndim <= 2:
                    np.savetxt(txt_file, data, fmt='%s')
                else:
                    # Flatten for saving, keeping original shape info at the top
                    flat_data = data.reshape(data.shape[0], -1)
                    for row in flat_data:
                        txt_file.write(' '.join(map(str, row)) + '\n')

            print(f"Saved dataset '{name}' to {txt_filename}")
        elif isinstance(obj, h5py.Group):
            print(f"Group: {name}")

    f.visititems(recursively_save)
