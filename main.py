from pypcd4.pypcd4 import PointCloud
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
from functools import reduce

def main(map_path, save_path, do_downsampling=False, voxel_size=0.1):
    #Parameters
    x_angle = 0
    y_tilt_angle = 180
    z_angle = 180

    #Read pcd map
    pc: PointCloud = PointCloud.from_path(map_path)
    points:np.ndarray = pc.numpy(["x", "y", "z", "intensity"])
    original_count = len(points)

    # Extract xyz and intensity
    xyz = points[:, :3].copy()
    intensity = points[:, 3].copy()

    # Downsampling
    if do_downsampling:
        voxel_coords = np.floor(xyz / voxel_size).astype(np.int32)
        _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
        xyz = xyz[unique_indices]
        intensity = intensity[unique_indices]
        print(f"Downsampled from {original_count} to {len(xyz)} points.")

    # Rotation matrix
    r = R.from_euler('xyz', [x_angle, y_tilt_angle, z_angle], degrees=True)
    rot_matrix = r.as_matrix().astype(np.float32)

    # Rotate only xyz while retaining intensity
    xyz_array = xyz.astype(np.float32)
    xyz_rotated = np.dot(xyz_array, rot_matrix.T).astype(np.float32)

    # Reshape intensity to (n, 1)
    intensity = intensity.reshape(-1, 1)
    xyzi = np.hstack((xyz_rotated, intensity))

    # Create new point cloud with rotated xyz and original intensity
    new_pc = PointCloud.from_xyzi_points(xyzi)
    new_pc.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and downsample a PCD map.")
    parser.add_argument("map_path", help="Path to input PCD map.")
    parser.add_argument("save_path", help="Path to save processed PCD map.")
    parser.add_argument("--downsample", action="store_true", default=False, help="Enable voxel downsampling.")
    args = parser.parse_args()
    main(args.map_path, args.save_path, do_downsampling=args.downsample)

