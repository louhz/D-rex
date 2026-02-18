

import numpy as np

dex_path = 'inference/cookie/grasps.npz'
data = np.load(dex_path)

# Gather joint angles j0 ... j15 in ascending order
joint_angles = np.array([data[f'j{i}'] for i in range(16)])

print("Joint angles from j0 to j15:", joint_angles[:,0])