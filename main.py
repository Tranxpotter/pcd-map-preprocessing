from pypcd4.pypcd4 import PointCloud
import numpy as np
from scipy.spatial.transform import Rotation as R
from functools import reduce

#Parameters
map_path = "../maps/IW-LG-scans-02.pcd"
save_path = "../maps/IW-LG-scans-02-processed2.pcd"
y_tilt_angle = -60
voxel_size = 0.1

#Read pcd map
pc: PointCloud = PointCloud.from_path(map_path)
points:np.ndarray = pc.numpy(("x", "y", "z"))
original_count = len(points)

#Rotation matrix
r = R.from_euler('y', y_tilt_angle, degrees=True)
rot_matrix = r.as_matrix().astype(np.float32)

# Downsampling
voxel_coords = np.floor(points / voxel_size).astype(np.int32)
# Find unique voxels and get their first index
_, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
points = points[unique_indices]
print(f"Downsampled from {original_count} to {len(points)} points.")

points_corrected = np.dot(points, rot_matrix.T).astype(np.float32)

new_pc = PointCloud.from_xyz_points(points_corrected)
new_pc.save(save_path)

