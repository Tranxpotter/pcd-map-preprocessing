from pypcd4.pypcd4 import PointCloud
import numpy as np
from scipy.spatial.transform import Rotation as R
from functools import reduce

#Parameters
map_path = "../maps/LG-scans-03.pcd"
save_path = "../maps/LG-scans-03-processed-01.pcd"
y_tilt_angle = 0
voxel_size = 0.1

#Read pcd map
pc: PointCloud = PointCloud.from_path(map_path)
points:np.ndarray = pc.numpy(["x", "y", "z", "intensity"])
original_count = len(points)

# Extract xyz and intensity
xyz = points[:, :3].copy()
intensity = points[:, 3].copy()
#Rotation matrix
r = R.from_euler('y', y_tilt_angle, degrees=True)
rot_matrix = r.as_matrix().astype(np.float32)

# Downsampling
voxel_coords = np.floor(xyz / voxel_size).astype(np.int32)
# Find unique voxels and get their first index
_, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
xyz = xyz[unique_indices]
intensity = intensity[unique_indices]
print(f"Downsampled from {original_count} to {len(xyz)} points.")

# Rotate only xyz while retaining intensity
xyz_array = np.column_stack([xyz[:, 0], xyz[:, 1], xyz[:, 2]]).astype(np.float32)
xyz_rotated = np.dot(xyz_array, rot_matrix.T).astype(np.float32)

# Reshape intensity to (n, 1)
intensity = intensity.reshape(-1, 1)
xyzi = np.hstack((xyz, intensity))

# Create new point cloud with rotated xyz and original intensity
new_pc = PointCloud.from_xyzi_points(xyzi)
new_pc.save(save_path)

