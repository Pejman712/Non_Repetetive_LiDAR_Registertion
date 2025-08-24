import numpy as np
import open3d as o3d
import os
import glob
from typing import List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def load_pcd_files(folder_path: str,
                   step_size: int = 1,
                   start_index: int = 0,
                   max_clouds: int = None,
                   voxel_size: float = 0.5,
                   apply_sor: bool = True,
                   sor_nb_neighbors: int = 50,
                   sor_std_ratio: float = 1.0) -> List[Tuple[str, o3d.geometry.PointCloud]]:
    """
    Load PCD files from a folder with optional downsampling and SOR filtering.

    Args:
        folder_path: Path to folder containing PCD files
        step_size: Load every Nth file (e.g., step_size=5 loads every 5th file)
        start_index: Starting index for sampling (0-based)
        max_clouds: Maximum number of clouds to load (None for no limit)
        voxel_size: Voxel size for downsampling (0 disables downsampling)
        apply_sor: Whether to apply Statistical Outlier Removal
        sor_nb_neighbors: Number of neighbors to analyze for SOR
        sor_std_ratio: Standard deviation multiplier for SOR
    Returns:
        List of (filename, pointcloud) tuples
    """
    pcd_files = glob.glob(os.path.join(folder_path, "*.pcd"))
    pcd_files.sort()  # Sort to ensure consistent ordering

    # Apply sampling
    sampled_files = pcd_files[start_index::step_size]

    # Apply max_clouds limit
    if max_clouds is not None and max_clouds > 0:
        sampled_files = sampled_files[:max_clouds]

    print(f"Found {len(pcd_files)} total PCD files")
    print(f"Sampling every {step_size} files starting from index {start_index}")
    if max_clouds is not None:
        print(f"Limited to first {max_clouds} clouds after sampling")
    print(f"Will process {len(sampled_files)} files: {[os.path.basename(f) for f in sampled_files[:5]]}{'...' if len(sampled_files) > 5 else ''}")

    point_clouds = []
    for file_path in sampled_files:
        try:
            pcd = o3d.io.read_point_cloud(file_path)
            if len(pcd.points) == 0:
                print(f"Warning: Empty point cloud in {file_path}")
                continue

            # Apply voxel downsampling
            if voxel_size > 0:
                pcd = pcd.voxel_down_sample(voxel_size)

            # Apply SOR filtering
            if apply_sor:
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=sor_nb_neighbors,
                                                        std_ratio=sor_std_ratio)

            point_clouds.append((os.path.basename(file_path), pcd))
            print(f"Loaded {file_path}: {len(pcd.points)} points")

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return point_clouds


def apply_gicp_wrapper(source_cloud, target_cloud, apply_gicp_func, voxel_size=0.05):
    """
    Wrapper function to adapt Open3D PointClouds for your apply_gicp function
    Args:
        source_cloud: Open3D PointCloud object
        target_cloud: Open3D PointCloud object
        apply_gicp_func: Your apply_gicp function
        voxel_size: voxel size parameter for your function
    Returns:
        4x4 transformation matrix
    """
    import numpy as np
    
    # Create simple wrapper class that mimics your expected input
    class PCDWrapper:
        def __init__(self, o3d_cloud):
            self.pcd = o3d_cloud
    
    # Wrap the Open3D clouds
    source_wrapped = PCDWrapper(source_cloud)
    target_wrapped = PCDWrapper(target_cloud)
    
    # Call your function
    return apply_gicp_func(source_wrapped, target_wrapped, voxel_size)

def apply_gicp_direct(source_cloud, target_cloud, voxel_size=0.05):
    """
    Direct implementation using small_gicp only
    Args:
        source_cloud: Open3D PointCloud object
        target_cloud: Open3D PointCloud object
        voxel_size: voxel size for downsampling (unused but kept for compatibility)
    Returns:
        4x4 transformation matrix
    """
    import numpy as np
    import small_gicp
    
    # Extract raw point data directly from Open3D PointClouds
    target_raw_numpy = np.asarray(target_cloud.points, dtype=np.float64)
    source_raw_numpy = np.asarray(source_cloud.points, dtype=np.float64)
    
    # Check if clouds have points
    if len(source_raw_numpy) == 0 or len(target_raw_numpy) == 0:
        print("Warning: Empty point cloud detected, returning identity matrix")
        return np.eye(4)
    
    # Perform alignment using small_gicp
    result = small_gicp.align(source_raw_numpy, target_raw_numpy)
    return result.T_target_source

def apply_gicp_open3d_fallback(source_cloud, target_cloud, voxel_size=0.05):
    """
    Fallback GICP implementation using Open3D's built-in ICP
    Args:
        source_cloud: Open3D PointCloud object
        target_cloud: Open3D PointCloud object
        voxel_size: voxel size for downsampling
    Returns:
        4x4 transformation matrix
    """
    import numpy as np
    import open3d as o3d
    
    try:
        # Create copies to avoid modifying originals
        source_copy = o3d.geometry.PointCloud(source_cloud)
        target_copy = o3d.geometry.PointCloud(target_cloud)
        
        # Downsample for efficiency
        if voxel_size > 0:
            source_copy = source_copy.voxel_down_sample(voxel_size)
            target_copy = target_copy.voxel_down_sample(voxel_size)
        
        # Estimate normals for better ICP
        source_copy.estimate_normals()
        target_copy.estimate_normals()
        
        # Initial alignment using global registration if available
        try:
            # Try FPFH-based global registration first
            radius_feature = voxel_size * 5
            source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                source_copy, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
            target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                target_copy, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
            
            # Global registration
            distance_threshold = voxel_size * 1.5
            global_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_copy, target_copy, source_fpfh, target_fpfh, True,
                distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3, [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
                ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
            
            initial_transform = global_result.transformation
        except:
            # Fallback to identity if global registration fails
            initial_transform = np.eye(4)
        
        # Refined ICP registration
        distance_threshold = voxel_size * 2.0
        icp_result = o3d.pipelines.registration.registration_icp(
            source_copy, target_copy, distance_threshold, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        
        if icp_result.fitness > 0.1:  # Reasonable fitness threshold
            return icp_result.transformation
        else:
            print(f"Warning: Low ICP fitness ({icp_result.fitness:.3f}), returning identity")
            return np.eye(4)
            
    except Exception as e:
        print(f"Error in Open3D ICP fallback: {e}, returning identity matrix")
        return np.eye(4)

