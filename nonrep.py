import numpy as np
import open3d as o3d
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import pandas as pd
import time

@dataclass
class ScanState:
    """State for non-repetitive LiDAR scan processing"""
    pose: np.ndarray  # [x, y, z, yaw]
    uncertainty: np.ndarray  # 4x4 covariance matrix
    confidence: float  # Confidence in this pose estimate
    scan_features: Dict  # Geometric features of the scan

class NonRepetitiveLiDARProcessor:
    def __init__(self, 
                 adaptive_threshold: float = 0.9,
                 feature_weight: float = 0.3,
                 geometric_weight: float = 0.4,
                 temporal_weight: float = 0.3,
                 force_z_zero: bool = False,
                 z_redistribution_method: str = 'prediction'):  # 'prediction', 'dominant_axis', 'equal'
        """
        Processor for non-repetitive LiDAR scans without velocity assumptions
        
        Args:
            adaptive_threshold: Threshold for switching prediction strategies
            feature_weight: Weight for feature-based matching
            geometric_weight: Weight for geometric consistency
            temporal_weight: Weight for temporal smoothness
            force_z_zero: If True, forces z coordinate to 0 and redistributes z values
            z_redistribution_method: Method for redistributing z values ('prediction', 'dominant_axis', 'equal')
        """
        self.adaptive_threshold = adaptive_threshold
        self.feature_weight = feature_weight
        self.geometric_weight = geometric_weight
        self.temporal_weight = temporal_weight
        self.force_z_zero = force_z_zero
        self.z_redistribution_method = z_redistribution_method
        
        # State tracking
        self.scan_states = []  # History of scan states
        self.feature_database = []  # Database of scan features
        self.motion_patterns = []  # Detected motion patterns
        
        # Adaptive parameters
        self.current_strategy = "feature_based"
        self.confidence_threshold = 0.7
        
        # Feature extraction parameters
        self.voxel_size = 0.1
        self.normal_radius = 0.5
        self.fpfh_radius = 1.0

    def redistribute_z_component(self, pose: np.ndarray, predicted_pose: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Redistribute z component to x and y coordinates based on the specified method
        
        Args:
            pose: Original pose [x, y, z, yaw]
            predicted_pose: Predicted pose for direction guidance
            
        Returns:
            Modified pose with z=0 and redistributed values
        """
        if not self.force_z_zero or abs(pose[2]) < 1e-6:
            return pose.copy()
        
        modified_pose = pose.copy()
        z_value = modified_pose[2]
        
        if self.z_redistribution_method == 'prediction' and predicted_pose is not None:
            # Use prediction to determine dominant movement direction
            if len(self.scan_states) >= 1:
                last_pose = self.scan_states[-1].pose
                #predicted_movement = predicted_pose[:3] - last_pose[:3]
                
                # Determine dominant movement axis based on prediction
                #abs_movement = np.abs(predicted_movement[:2])  # Only x and y
                #if abs_movement[0] > abs_movement[1]:
                    # X movement is dominant
                #modified_pose[0] += 1 * -z_value + predicted_movement[0]
            """
                    print(f"Redistributing z={z_value:.3f} to x based on prediction")
                #else:
                #    # Y movement is dominant
                #    modified_pose[1] += 0.1 * z_value + predicted_movement[1]
                #    print(f"Redistributing z={z_value:.3f} to y based on prediction")
            #else:
                # Fallback to equal distribution for first scan
               # modified_pose[0] += z_value * 0.5
               # modified_pose[1] += z_value * 0.5
               # print(f"Redistributing z={z_value:.3f} equally to x and y (first scan)")
                
        elif self.z_redistribution_method == 'dominant_axis':
            # Use historical movement to determine dominant axis
            if len(self.scan_states) >= 2:
                recent_poses = [state.pose for state in self.scan_states[-3:]]
                x_movements = []
                y_movements = []
                
                for i in range(1, len(recent_poses)):
                    x_movements.append(abs(recent_poses[i][0] - recent_poses[i-1][0]))
                    y_movements.append(abs(recent_poses[i][1] - recent_poses[i-1][1]))
                
                avg_x_movement = np.mean(x_movements)
                avg_y_movement = np.mean(y_movements)
                
                if avg_x_movement > avg_y_movement:
                    modified_pose[0] += z_value
                    print(f"Redistributing z={z_value:.3f} to x based on dominant axis")
                else:
                    modified_pose[1] += z_value
                    print(f"Redistributing z={z_value:.3f} to y based on dominant axis")
            else:
                # Equal distribution for early scans
                modified_pose[0] += z_value * 0.5
                modified_pose[1] += z_value * 0.5
                print(f"Redistributing z={z_value:.3f} equally (insufficient history)")
                
        elif self.z_redistribution_method == 'equal':
            # Distribute equally between x and y
            modified_pose[0] += z_value * 0.5
            modified_pose[1] += z_value * 0.5
            print(f"Redistributing z={z_value:.3f} equally to x and y")
        """
        # Force z to zero
        modified_pose[2] = 0.0
        
        return modified_pose

    def extract_scan_features(self, cloud: o3d.geometry.PointCloud) -> Dict:
        """
        Extract geometric features from LiDAR scan for non-repetitive matching
        
        Args:
            cloud: Input point cloud
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        try:
            # Basic geometric properties
            points = np.asarray(cloud.points)
            if len(points) == 0:
                return features
            
            # 1. Statistical features
            features['point_count'] = len(points)
            features['centroid'] = np.mean(points, axis=0)
            features['std_dev'] = np.std(points, axis=0)
            features['bounding_box'] = {
                'min': np.min(points, axis=0),
                'max': np.max(points, axis=0),
                'extent': np.max(points, axis=0) - np.min(points, axis=0)
            }
            
            # 2. Downsampling for feature extraction
            if len(points) > 1000:
                cloud_ds = cloud.voxel_down_sample(self.voxel_size)
            else:
                cloud_ds = cloud
            
            # 3. Normal estimation
            if len(cloud_ds.points) > 10:
                cloud_ds.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=self.normal_radius, max_nn=30
                    )
                )
                
                normals = np.asarray(cloud_ds.normals)
                if len(normals) > 0:
                    features['normal_distribution'] = {
                        'mean': np.mean(normals, axis=0),
                        'std': np.std(normals, axis=0)
                    }
            
            # 4. FPFH features for distinctive geometric signatures
            if len(cloud_ds.points) > 50 and cloud_ds.has_normals():
                fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    cloud_ds,
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=self.fpfh_radius, max_nn=100
                    )
                )
                features['fpfh_histogram'] = np.asarray(fpfh.data).mean(axis=1)
            
            # 5. Planar structures detection
            if len(cloud_ds.points) > 100:
                plane_model, inliers = cloud_ds.segment_plane(
                    distance_threshold=0.1,
                    ransac_n=3,
                    num_iterations=1000
                )
                
                if len(inliers) > 50:
                    features['dominant_plane'] = {
                        'normal': plane_model[:3],
                        'distance': plane_model[3],
                        'inlier_ratio': len(inliers) / len(cloud_ds.points)
                    }
            
            # 6. Height distribution (for outdoor LiDAR)
            z_coords = points[:, 2]
            features['height_profile'] = {
                'min_height': np.min(z_coords),
                'max_height': np.max(z_coords),
                'mean_height': np.mean(z_coords),
                'height_variance': np.var(z_coords)
            }
            
            # 7. Density analysis
            if len(points) > 100:
                # Sample points for density estimation
                sample_indices = np.random.choice(len(points), min(100, len(points)), replace=False)
                sample_points = points[sample_indices]
                
                distances = cdist(sample_points, points)
                k_nearest_dists = np.sort(distances, axis=1)[:, 1:6]  # 5 nearest neighbors
                avg_density = np.mean(k_nearest_dists)
                features['local_density'] = avg_density
            
            # 8. Shape complexity
            if len(points) > 20:
                # PCA analysis for shape understanding
                pca = PCA(n_components=3)
                pca.fit(points)
                features['shape_complexity'] = {
                    'explained_variance_ratio': pca.explained_variance_ratio_,
                    'linearity': pca.explained_variance_ratio_[0],
                    'planarity': pca.explained_variance_ratio_[1],
                    'sphericity': pca.explained_variance_ratio_[2]
                }
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            features['extraction_error'] = str(e)
        
        return features

    def compute_feature_similarity(self, features1: Dict, features2: Dict) -> float:
        """
        Compute similarity between two feature sets
        
        Args:
            features1, features2: Feature dictionaries
            
        Returns:
            Similarity score [0, 1]
        """
        if not features1 or not features2:
            return 0.0
        
        similarities = []
        
        try:
            # 1. Point count similarity
            if 'point_count' in features1 and 'point_count' in features2:
                count_ratio = min(features1['point_count'], features2['point_count']) / \
                             max(features1['point_count'], features2['point_count'])
                similarities.append(count_ratio)
            
            # 2. Centroid distance (normalized) - only consider x,y if force_z_zero is True
            if 'centroid' in features1 and 'centroid' in features2:
                if self.force_z_zero:
                    centroid_dist = np.linalg.norm(features1['centroid'][:2] - features2['centroid'][:2])
                else:
                    centroid_dist = np.linalg.norm(features1['centroid'] - features2['centroid'])
                # Normalize by typical scan range (assume 50m max)
                centroid_sim = max(0, 1 - centroid_dist / 50.0)
                similarities.append(centroid_sim)
            
            # 3. Bounding box similarity
            if ('bounding_box' in features1 and 'bounding_box' in features2):
                bb1, bb2 = features1['bounding_box'], features2['bounding_box']
                if self.force_z_zero:
                    # Only consider x,y extents
                    extent1 = bb1['extent'][:2]
                    extent2 = bb2['extent'][:2]
                else:
                    extent1 = bb1['extent']
                    extent2 = bb2['extent']
                extent_ratio = np.prod(np.minimum(extent1, extent2)) / \
                              np.prod(np.maximum(extent1, extent2))
                similarities.append(extent_ratio)
            
            # 4. FPFH feature similarity
            if ('fpfh_histogram' in features1 and 'fpfh_histogram' in features2):
                fpfh1, fpfh2 = features1['fpfh_histogram'], features2['fpfh_histogram']
                if len(fpfh1) == len(fpfh2):
                    # Cosine similarity
                    dot_product = np.dot(fpfh1, fpfh2)
                    norm_product = np.linalg.norm(fpfh1) * np.linalg.norm(fpfh2)
                    if norm_product > 0:
                        fpfh_sim = dot_product / norm_product
                        similarities.append(max(0, fpfh_sim))
            
            # 5. Height profile similarity (modified for z=0 mode)
            if ('height_profile' in features1 and 'height_profile' in features2):
                hp1, hp2 = features1['height_profile'], features2['height_profile']
                if not self.force_z_zero:
                    height_range_ratio = min(hp1['max_height'] - hp1['min_height'],
                                           hp2['max_height'] - hp2['min_height']) / \
                                       max(hp1['max_height'] - hp1['min_height'],
                                           hp2['max_height'] - hp2['min_height'])
                    similarities.append(height_range_ratio)
                else:
                    # In z=0 mode, consider height variance instead of range
                    var_ratio = min(hp1['height_variance'], hp2['height_variance']) / \
                               max(hp1['height_variance'], hp2['height_variance'])
                    similarities.append(var_ratio)
            
            # 6. Density similarity
            if ('local_density' in features1 and 'local_density' in features2):
                density_ratio = min(features1['local_density'], features2['local_density']) / \
                               max(features1['local_density'], features2['local_density'])
                similarities.append(density_ratio)
            
        except Exception as e:
            print(f"Error computing similarity: {e}")
        
        # Return weighted average of similarities
        if similarities:
            return np.mean(similarities)
        else:
            return 0.0

    def predict_pose_feature_based(self, current_features: Dict) -> Tuple[np.ndarray, float]:
        """
        Predict pose based on feature matching with previous scans
        
        Args:
            current_features: Features of current scan
            
        Returns:
            (predicted_pose, confidence)
        """
        if len(self.feature_database) < 2:
            return None, 0.0
        
        # Find most similar previous scans
        similarities = []
        for i, (features, state) in enumerate(zip(self.feature_database, self.scan_states)):
            sim = self.compute_feature_similarity(current_features, features)
            similarities.append((sim, i, state))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        if len(similarities) < 2:
            return None, 0.0
        
        # Use top matches for prediction
        best_sim, best_idx, best_state = similarities[0]
        second_sim, second_idx, second_state = similarities[1]
        
        if best_sim < 0.3:  # Low similarity threshold
            return None, 0.0
        
        # Weighted prediction based on similarity
        weight1 = best_sim / (best_sim + second_sim)
        weight2 = second_sim / (best_sim + second_sim)
        
        predicted_pose = weight1 * best_state.pose + weight2 * second_state.pose
        confidence = (best_sim + second_sim) / 2
        
        return predicted_pose, confidence

    def predict_pose_geometric_consistency(self) -> Tuple[np.ndarray, float]:
        """
        Predict pose based on geometric consistency with recent scans
        
        Returns:
            (predicted_pose, confidence)
        """
        if len(self.scan_states) < 3:
            return None, 0.0
        
        # Use last 3-5 poses for geometric consistency
        recent_states = self.scan_states[-min(5, len(self.scan_states)):]
        poses = [state.pose for state in recent_states]
        
        # Fit smooth trajectory through recent poses
        if len(poses) >= 3:
            # Simple polynomial fit for each dimension
            predicted_pose = np.zeros(4)
            confidence_scores = []
            
            # Handle z coordinate based on force_z_zero setting
            dimensions_to_predict = 4 if not self.force_z_zero else [0, 1, 3]  # Skip z if forced to zero
            
            for dim in dimensions_to_predict if self.force_z_zero else range(4):
                values = [pose[dim] for pose in poses]
                x = np.arange(len(values))
                
                # Fit polynomial (degree depends on number of points)
                degree = min(2, len(values) - 1)
                if degree > 0:
                    coeffs = np.polyfit(x, values, degree)
                    next_x = len(values)
                    predicted_pose[dim] = np.polyval(coeffs, next_x)
                    
                    # Estimate confidence based on fit quality
                    fitted_values = np.polyval(coeffs, x)
                    residuals = np.array(values) - fitted_values
                    mse = np.mean(residuals**2)
                    confidence_scores.append(max(0, 1 - mse))
                else:
                    predicted_pose[dim] = values[-1]
                    confidence_scores.append(0.5)
            
            # Force z to zero if required
            if self.force_z_zero:
                predicted_pose[2] = 0.0
                
            # Handle angle wrapping for yaw
            predicted_pose[3] = np.arctan2(np.sin(predicted_pose[3]), np.cos(predicted_pose[3]))
            
            avg_confidence = np.mean(confidence_scores)
            return predicted_pose, avg_confidence
        
        return None, 0.0

    def predict_pose_adaptive(self, current_features: Dict) -> Tuple[np.ndarray, float]:
        """
        Adaptive pose prediction combining multiple strategies
        
        Args:
            current_features: Features of current scan
            
        Returns:
            (predicted_pose, confidence)
        """
        predictions = []
        
        # 1. Feature-based prediction
        feature_pose, feature_conf = self.predict_pose_feature_based(current_features)
        if feature_pose is not None:
            predictions.append((feature_pose, feature_conf, 'feature'))
        
        # 2. Geometric consistency prediction
        geom_pose, geom_conf = self.predict_pose_geometric_consistency()
        if geom_pose is not None:
            predictions.append((geom_pose, geom_conf, 'geometric'))
        
        # 3. Simple extrapolation (fallback)
        if len(self.scan_states) >= 2:
            last_pose = self.scan_states[-1].pose
            prev_pose = self.scan_states[-2].pose
            extrapolated_pose = last_pose + 0.3 * (last_pose - prev_pose)  # Damped extrapolation
            extrapolated_pose[3] = np.arctan2(np.sin(extrapolated_pose[3]), np.cos(extrapolated_pose[3]))
            
            # Force z to zero if required
            if self.force_z_zero:
                extrapolated_pose[2] = 0.0
                
            predictions.append((extrapolated_pose, 0.4, 'extrapolation'))
        
        if not predictions:
            return None, 0.0
        
        # Adaptive fusion based on confidence and strategy performance
        if len(predictions) == 1:
            return predictions[0][0], predictions[0][1]
        
        # Weight predictions by confidence and strategy reliability
        total_weight = 0
        weighted_pose = np.zeros(4)
        
        for pose, conf, strategy in predictions:
            # Strategy-specific weights
            if strategy == 'feature':
                strategy_weight = self.feature_weight
            elif strategy == 'geometric':
                strategy_weight = self.geometric_weight
            else:
                strategy_weight = self.temporal_weight
            
            weight = conf * strategy_weight
            weighted_pose += weight * pose
            total_weight += weight
        
        if total_weight > 0:
            final_pose = weighted_pose / total_weight
            final_confidence = total_weight / len(predictions)
            
            # Force z to zero if required
            if self.force_z_zero:
                final_pose[2] = 0.0
                
            return final_pose, final_confidence
        
        return None, 0.0

    def update_with_observation(self, 
                              observed_pose: np.ndarray, 
                              scan_features: Dict,
                              registration_confidence: float = 1.0,
                              predicted_pose: Optional[np.ndarray] = None):
        """
        Update state with new observation
        
        Args:
            observed_pose: Observed pose from registration
            scan_features: Features of the scan
            registration_confidence: Confidence in the registration
            predicted_pose: Predicted pose for z redistribution guidance
        """
        # Apply z redistribution if required
        final_pose = self.redistribute_z_component(observed_pose, predicted_pose)
        
        # Create new scan state
        # Simple uncertainty model - could be made more sophisticated
        base_uncertainty = 0.1
        uncertainty_matrix = np.eye(4) * (base_uncertainty / registration_confidence) ** 2
        
        new_state = ScanState(
            pose=final_pose.copy(),
            uncertainty=uncertainty_matrix,
            confidence=registration_confidence,
            scan_features=scan_features
        )
        
        # Add to history
        self.scan_states.append(new_state)
        self.feature_database.append(scan_features)
        
        # Keep limited history
        max_history = 20
        if len(self.scan_states) > max_history:
            self.scan_states.pop(0)
            self.feature_database.pop(0)
        
        # Analyze motion patterns
        self._analyze_motion_patterns()

    def _analyze_motion_patterns(self):
        """Analyze motion patterns for adaptive strategy selection"""
        if len(self.scan_states) < 5:
            return
        
        # Analyze recent motion characteristics
        recent_poses = [state.pose for state in self.scan_states[-5:]]
        
        # Calculate motion metrics
        position_changes = []
        yaw_changes = []
        
        for i in range(1, len(recent_poses)):
            if self.force_z_zero:
                # Only consider x,y movement
                pos_change = np.linalg.norm(recent_poses[i][:2] - recent_poses[i-1][:2])
            else:
                pos_change = np.linalg.norm(recent_poses[i][:3] - recent_poses[i-1][:3])
            
            yaw_change = abs(recent_poses[i][3] - recent_poses[i-1][3])
            position_changes.append(pos_change)
            yaw_changes.append(yaw_change)
        
        # Classify motion pattern
        avg_pos_change = np.mean(position_changes)
        std_pos_change = np.std(position_changes)
        
        if avg_pos_change < 0.1:
            pattern = "stationary"
        elif std_pos_change / (avg_pos_change + 1e-6) < 0.3:
            pattern = "smooth"
        elif std_pos_change / (avg_pos_change + 1e-6) > 1.0:
            pattern = "erratic"
        else:
            pattern = "variable"
        
        self.motion_patterns.append(pattern)
        
        # Keep limited pattern history
        if len(self.motion_patterns) > 10:
            self.motion_patterns.pop(0)
        
        # Adapt strategy weights based on patterns
        if pattern == "erratic":
            self.feature_weight = 0.5  # Rely more on features
            self.geometric_weight = 0.2
            self.temporal_weight = 0.3
        elif pattern == "smooth":
            self.feature_weight = 0.2
            self.geometric_weight = 0.5  # Rely more on geometric consistency
            self.temporal_weight = 0.3
        else:  # stationary or variable
            self.feature_weight = 0.35
            self.geometric_weight = 0.35
            self.temporal_weight = 0.3

    def get_current_state(self) -> Optional[ScanState]:
        """Get current scan state"""
        if self.scan_states:
            return self.scan_states[-1]
        return None

    def get_motion_analysis(self) -> Dict:
        """Get motion analysis summary"""
        if len(self.scan_states) < 2:
            return {}
        
        poses = [state.pose for state in self.scan_states]
        confidences = [state.confidence for state in self.scan_states]
        
        # Calculate statistics
        position_moves = []
        for i in range(1, len(poses)):
            if self.force_z_zero:
                move = np.linalg.norm(poses[i][:2] - poses[i-1][:2])
            else:
                move = np.linalg.norm(poses[i][:3] - poses[i-1][:3])
            position_moves.append(move)
        
        analysis = {
            'scan_count': len(self.scan_states),
            'avg_movement': np.mean(position_moves) if position_moves else 0,
            'movement_std': np.std(position_moves) if position_moves else 0,
            'avg_confidence': np.mean(confidences),
            'recent_pattern': self.motion_patterns[-1] if self.motion_patterns else 'unknown',
            'current_weights': {
                'feature': self.feature_weight,
                'geometric': self.geometric_weight,
                'temporal': self.temporal_weight
            },
            'force_z_zero': self.force_z_zero,
            'z_redistribution_method': self.z_redistribution_method
        }
        
        return analysis

def process_non_repetitive_lidar_scans(observation_folder: str,
                                     apply_gicp_func=None,
                                     visualize=True,
                                     observation_step_size=1,
                                     observation_start_index=0,
                                     max_observation_clouds=None,
                                     force_z_zero=False,
                                     z_redistribution_method='prediction',
                                     # --- NEW: ablation controls ---
                                     fixed_weights: Optional[Tuple[float, float, float]] = None,
                                     freeze_adaptation: bool = False):
    """
    Process non-repetitive LiDAR scans

    Args:
        observation_folder: Path to observation PCD files
        apply_gicp_func: GICP function for registration
        visualize: Whether to show visualizations
        observation_step_size: Load every Nth observation file
        observation_start_index: Starting index for observation sampling
        max_observation_clouds: Maximum number of clouds to process
        force_z_zero: If True, forces z coordinate to 0 and redistributes z values
        z_redistribution_method: Method for redistributing z values ('prediction', 'dominant_axis', 'equal')

        fixed_weights: Optional (feature, geometric, temporal). If provided, the processor
                       will use these fusion weights for prediction.
        freeze_adaptation: If True, disables internal adaptive re-weighting so weights remain fixed.
    """
    # Load clouds
    if apply_gicp_func is None:
        from Pctools import apply_gicp_direct
        apply_gicp_func = apply_gicp_direct

    from Pctools import load_pcd_files
    observation_pcds = load_pcd_files(observation_folder, observation_step_size,
                                      observation_start_index, max_observation_clouds)

    if len(observation_pcds) == 0:
        print("Error: No observation point clouds found")
        return

    print(f"\n=== Non-Repetitive LiDAR Processing ===")
    print(f"Processing {len(observation_pcds)} scans")
    print(f"Adaptive feature-based prediction enabled")
    print(f"Force Z=0: {force_z_zero}")
    if force_z_zero:
        print(f"Z redistribution method: {z_redistribution_method}")

    # Initialize processor
    processor = NonRepetitiveLiDARProcessor(
        force_z_zero=force_z_zero,
        z_redistribution_method=z_redistribution_method
    )

    # --- NEW: apply ablation controls ---
    if fixed_weights is not None:
        fw, gw, tw = fixed_weights
        processor.feature_weight = float(fw)
        processor.geometric_weight = float(gw)
        processor.temporal_weight = float(tw)
        print(f"[Ablation] Fixed weights -> feature={fw}, geometric={gw}, temporal={tw}")

    if freeze_adaptation:
        # Disable adaptive reweighting by making the analyzer a no-op
        processor._analyze_motion_patterns = lambda *args, **kwargs: None
        print("[Ablation] Adaptive re-weighting is FROZEN (no strategy updates).")

    # Store results
    results = []
    predicted_poses = []
    observed_poses = []
    final_poses = []

    cumulative_transform = None

    for i, (scan_name, scan_cloud) in enumerate(observation_pcds):
        print(f"\n=== Processing scan {i+1}/{len(observation_pcds)}: {scan_name} ===")

        try:
            # Extract features from current scan
            print("Extracting scan features...")
            current_features = processor.extract_scan_features(scan_cloud)

            # Predict pose using adaptive method (weights may be frozen/fixed)
            print("Predicting pose...")
            predicted_pose, prediction_confidence = processor.predict_pose_adaptive(current_features)
            predicted_poses.append(predicted_pose.copy() if predicted_pose is not None else None)

            if predicted_pose is not None:
                print(f"Predicted pose: {predicted_pose} (confidence: {prediction_confidence:.3f})")
            else:
                print("No prediction available")

            # GICP registration for observation
            observed_pose = None
            if i < len(observation_pcds) - 1:
                next_scan_name, next_scan_cloud = observation_pcds[i + 1]
                print(f"GICP registration: {scan_name} -> {next_scan_name}")

                # Get transformation from GICP
                transformation = apply_gicp_func(scan_cloud, next_scan_cloud)

                # Accumulate transformation
                if cumulative_transform is not None:
                    cumulative_transform = cumulative_transform @ transformation
                else:
                    cumulative_transform = transformation

                # Convert to pose
                observed_pose = transformation_to_pose(cumulative_transform)
                print(f"Observed pose (before z redistribution): {observed_pose}")

                # Estimate registration confidence (simple heuristic)
                reg_confidence = estimate_registration_confidence(scan_cloud, next_scan_cloud, transformation)
                print(f"Registration confidence: {reg_confidence:.3f}")

                # Update processor with observation (includes z redistribution)
                processor.update_with_observation(observed_pose, current_features, reg_confidence, predicted_pose)

                # Get the final pose after z redistribution
                current_state = processor.get_current_state()
                if current_state is not None:
                    final_observed_pose = current_state.pose
                    print(f"Final observed pose (after z redistribution): {final_observed_pose}")
                else:
                    final_observed_pose = observed_pose

            else:
                print("Last scan - no registration")
                if len(observed_poses) > 0 and observed_poses[-1] is not None:
                    observed_pose = observed_poses[-1].copy()
                else:
                    observed_pose = np.array([0.0, 0.0, 0.0, 0.0])
                final_observed_pose = observed_pose

            observed_poses.append(final_observed_pose.copy() if final_observed_pose is not None else None)

            # Get final pose estimate
            current_state = processor.get_current_state()
            if current_state is not None:
                final_pose = current_state.pose
                final_poses.append(final_pose.copy())
                print(f"Final pose: {final_pose}")
            else:
                final_poses.append(None)
                print("No final pose estimate")

            # Store results
            motion_analysis = processor.get_motion_analysis()

            result = {
                'scan_file': scan_name,
                'scan_index': observation_start_index + i * observation_step_size,
                'predicted_pose': predicted_pose.copy() if predicted_pose is not None else None,
                'observed_pose': final_observed_pose.copy() if final_observed_pose is not None else None,
                'final_pose': final_pose.copy() if current_state is not None else None,
                'prediction_confidence': prediction_confidence if predicted_pose is not None else 0.0,
                'features': current_features,
                'motion_analysis': motion_analysis,
                'processing_method': 'ablation_fixed_weights' if fixed_weights else (
                    'non_repetitive_adaptive_z_zero' if force_z_zero else 'non_repetitive_adaptive'
                ),
                'z_redistributed': force_z_zero,
                'weights': {
                    'feature': processor.feature_weight,
                    'geometric': processor.geometric_weight,
                    'temporal': processor.temporal_weight
                },
                'adaptation_frozen': freeze_adaptation
            }
            results.append(result)

        except Exception as e:
            print(f"Error processing {scan_name}: {e}")
            import traceback
            traceback.print_exc()

            results.append({
                'scan_file': scan_name,
                'error': str(e)
            })
            predicted_poses.append(None)
            observed_poses.append(None)
            final_poses.append(None)

    # Final analysis
    motion_analysis = processor.get_motion_analysis()
    print(f"\n=== Processing Summary ===")
    print(f"Successfully processed: {len([r for r in results if 'error' not in r])} scans")
    if motion_analysis:
        print(f"Average movement: {motion_analysis['avg_movement']:.3f} m")
        print(f"Average confidence: {motion_analysis['avg_confidence']:.3f}")
        print(f"Motion pattern: {motion_analysis['recent_pattern']}")
        print(f"Strategy weights: {motion_analysis['current_weights']}")
        print(f"Z-coordinate handling: {'Forced to zero' if motion_analysis['force_z_zero'] else 'Natural'}")
        if motion_analysis['force_z_zero']:
            print(f"Z redistribution method: {motion_analysis['z_redistribution_method']}")

    # Visualization
    if visualize and len(final_poses) > 0:
        combined_cloud = create_combined_cloud(observation_pcds, final_poses)
        if combined_cloud is not None:
            print("\nOpening combined LiDAR map...")
            window_name = "Non-Repetitive LiDAR Map (Z=0)" if force_z_zero else "Non-Repetitive LiDAR Map"
            o3d.visualization.draw_geometries([combined_cloud],
                                              window_name=window_name,
                                              width=1400, height=900)

        plot_non_repetitive_analysis(predicted_poses, observed_poses, final_poses, processor)

    return results, predicted_poses, observed_poses, final_poses


def transformation_to_pose(transformation_matrix: np.ndarray) -> np.ndarray:
    """Convert 4x4 transformation matrix to pose [x,y,z,yaw]"""
    translation = transformation_matrix[:3, 3]
    rotation_matrix = transformation_matrix[:3, :3]
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return np.array([translation[0], translation[1], translation[2], yaw])

def estimate_registration_confidence(cloud1, cloud2, transformation, sample_size=100):
    """Estimate confidence in GICP registration result"""
    try:
        # Sample points for confidence estimation
        points1 = np.asarray(cloud1.points)
        points2 = np.asarray(cloud2.points)
        
        if len(points1) == 0 or len(points2) == 0:
            return 0.1
        
        # Sample subset of points
        sample_indices = np.random.choice(len(points1), min(sample_size, len(points1)), replace=False)
        sample_points = points1[sample_indices]
        
        # Transform sample points
        sample_points_hom = np.column_stack([sample_points, np.ones(len(sample_points))])
        transformed_points = (transformation @ sample_points_hom.T).T[:, :3]
        
        # Find nearest neighbors in target cloud
        from scipy.spatial import cKDTree
        tree = cKDTree(points2)
        distances, _ = tree.query(transformed_points)
        
        # Confidence based on average nearest neighbor distance
        avg_distance = np.mean(distances)
        confidence = max(0.1, min(1.0, 1.0 - avg_distance / 2.0))  # Normalize to [0.1, 1.0]
        
        return confidence
        
    except Exception as e:
        print(f"Error estimating confidence: {e}")
        return 0.5

def create_combined_cloud(observation_pcds, final_poses):
    """Create combined point cloud from poses"""
    if not observation_pcds or not final_poses:
        return None
    
    combined_cloud = o3d.geometry.PointCloud()
    
    for i, ((scan_name, scan_cloud), pose) in enumerate(zip(observation_pcds, final_poses)):
        if pose is None:
            continue
        
        if i == 0:
            # First cloud at origin
            combined_cloud += scan_cloud
        else:
            # Transform subsequent clouds
            T = np.eye(4)
            T[0, 3] = pose[0]  # x
            T[1, 3] = pose[1]  # y
            T[2, 3] = pose[2]  # z (will be 0 if force_z_zero is True)
            
            # Yaw rotation
            yaw = pose[3]
            T[0, 0] = np.cos(yaw)
            T[0, 1] = -np.sin(yaw)
            T[1, 0] = np.sin(yaw)
            T[1, 1] = np.cos(yaw)
            
            transformed_cloud = o3d.geometry.PointCloud(scan_cloud)
            transformed_cloud.transform(T)
            combined_cloud += transformed_cloud
    
    return combined_cloud

def plot_non_repetitive_analysis(predicted_poses, observed_poses, final_poses, processor, save_path="movement_magnitude.png"):
    """Plot ONLY the Movement Magnitude analysis of non-repetitive LiDAR processing.
    
    - Filters valid final poses
    - Computes per-scan movement (XY if Z=0 mode, else XYZ)
    - Plots a square chart: Movement Magnitude over scan index
    - Saves figure in high quality (default: PNG, 300 DPI)
    - Prints detailed analysis after plotting
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # 1) Filter valid poses
    valid_indices = [i for i in range(len(final_poses)) if final_poses[i] is not None]
    if not valid_indices:
        print("No valid poses to plot")
        return

    valid_final = [final_poses[i] for i in valid_indices]
    final_array = np.array(valid_final)
    time_steps = np.arange(len(valid_final))

    # 2) Motion analysis values (if available)
    motion_analysis = processor.get_motion_analysis() if hasattr(processor, "get_motion_analysis") else None

    # 3) Compute movement magnitudes between consecutive valid poses
    movements = []
    for i in range(1, len(valid_final)):
        if getattr(processor, "force_z_zero", False):
            # XY only
            movement = np.linalg.norm(final_array[i, :2] - final_array[i - 1, :2])
        else:
            # XYZ
            movement = np.linalg.norm(final_array[i, :3] - final_array[i - 1, :3])
        movements.append(movement)

    if not movements:
        print("Not enough valid poses to compute movement (need at least 2).")
        return

    movement_time = time_steps[1:]
    movement_label = 'Movement (m) - XY only' if getattr(processor, "force_z_zero", False) else 'Movement (m)'

    # 4) Square figure for the last plot only
    fig, ax_features = plt.subplots(figsize=(8, 8))  # square figure

    ax_features.plot(movement_time, movements, 'purple', linewidth=2, marker='d', markersize=5, label='Movement')

    # Optional average line if available
    if motion_analysis and isinstance(motion_analysis, dict) and 'avg_movement' in motion_analysis:
        ax_features.axhline(
            y=motion_analysis['avg_movement'],
            color='orange',
            linestyle='--',
            label=f"Avg: {motion_analysis['avg_movement']:.3f}m"
        )

    # 5) Styling / fonts
    ax_features.set_xlabel('Scan Index', fontsize=20)
    ax_features.set_ylabel(movement_label, fontsize=20)
    ax_features.set_title('Movement Magnitude', fontsize=20)
    ax_features.legend(fontsize=16)
    ax_features.tick_params(axis='both', labelsize=16)
    ax_features.grid(True, alpha=0.3)

    plt.tight_layout()

    # 6) Save high-quality figure
    plt.savefig(save_path, dpi=900, bbox_inches='tight')  # High-res save
    print(f"Plot saved as: {save_path}")

    plt.show()

    # 7) Keep your detailed analysis printout (non-visual)
    try:
        print_detailed_analysis(final_array, processor)
    except NameError:
        # If helper isn't defined in this scope, skip gracefully.
        pass


def print_detailed_analysis(final_array, processor):
    """Print detailed analysis of the processing results"""
    print(f"\n=== Detailed Analysis ===")
    
    if len(final_array) > 1:
        # Calculate trajectory statistics
        total_distance = 0
        movements = []
        yaw_changes = []
        z_redistributions = 0
        
        for i in range(1, len(final_array)):
            if processor.force_z_zero:
                movement = np.linalg.norm(final_array[i, :2] - final_array[i-1, :2])
            else:
                movement = np.linalg.norm(final_array[i, :3] - final_array[i-1, :3])
            movements.append(movement)
            total_distance += movement
            
            yaw_change = abs(final_array[i, 3] - final_array[i-1, 3])
            if yaw_change > np.pi:
                yaw_change = 2*np.pi - yaw_change
            yaw_changes.append(yaw_change)
        
        print(f"Trajectory Statistics:")
        if processor.force_z_zero:
            print(f"  Mode: 2D trajectory (Z forced to 0)")
            print(f"  Z redistribution method: {processor.z_redistribution_method}")
        else:
            print(f"  Mode: 3D trajectory")
        print(f"  Total distance: {total_distance:.3f} m")
        print(f"  Average movement per scan: {np.mean(movements):.3f} m")
        print(f"  Movement std deviation: {np.std(movements):.3f} m")
        print(f"  Max single movement: {np.max(movements):.3f} m")
        print(f"  Total yaw change: {np.degrees(np.sum(yaw_changes)):.1f}°")
        print(f"  Average yaw change: {np.degrees(np.mean(yaw_changes)):.1f}°")
        
        # Z coordinate analysis
        if processor.force_z_zero:
            z_values = final_array[:, 2]
            print(f"  Z coordinate verification: all values = {np.unique(z_values)}")
        else:
            z_range = np.max(final_array[:, 2]) - np.min(final_array[:, 2])
            print(f"  Z coordinate range: {z_range:.3f} m")
        
        # Movement consistency
        cv = np.std(movements) / (np.mean(movements) + 1e-6)
        print(f"  Movement consistency (CV): {cv:.3f}")
        
        if cv < 0.3:
            consistency = "Very Consistent"
        elif cv < 0.6:
            consistency = "Moderately Consistent" 
        elif cv < 1.0:
            consistency = "Variable"
        else:
            consistency = "Highly Variable"
        
        print(f"  Motion pattern: {consistency}")
    
    # Processor analysis
    motion_analysis = processor.get_motion_analysis()
    if motion_analysis:
        print(f"\nAdaptive Processing Analysis:")
        print(f"  Scans processed: {motion_analysis['scan_count']}")
        print(f"  Average confidence: {motion_analysis['avg_confidence']:.3f}")
        print(f"  Recent motion pattern: {motion_analysis['recent_pattern']}")
        print(f"  Current strategy weights:")
        for strategy, weight in motion_analysis['current_weights'].items():
            print(f"    {strategy.capitalize()}: {weight:.2f}")
    
    # Feature extraction summary
    feature_stats = analyze_feature_database(processor.feature_database)
    if feature_stats:
        print(f"\nFeature Extraction Summary:")
        for stat_name, value in feature_stats.items():
            print(f"  {stat_name}: {value}")

def analyze_feature_database(feature_database):
    """Analyze the collected feature database"""
    if not feature_database:
        return {}
    
    stats = {}
    
    # Count successful feature extractions
    successful_extractions = sum(1 for features in feature_database if features and 'extraction_error' not in features)
    stats['Successful extractions'] = f"{successful_extractions}/{len(feature_database)}"
    
    # Analyze point counts
    point_counts = [features.get('point_count', 0) for features in feature_database if features]
    if point_counts:
        stats['Avg points per scan'] = f"{np.mean(point_counts):.0f}"
        stats['Point count range'] = f"{np.min(point_counts):.0f} - {np.max(point_counts):.0f}"
    
    # Analyze FPFH availability
    fpfh_available = sum(1 for features in feature_database if features and 'fpfh_histogram' in features)
    stats['FPFH features available'] = f"{fpfh_available}/{len(feature_database)}"
    
    # Analyze plane detection
    planes_detected = sum(1 for features in feature_database if features and 'dominant_plane' in features)
    stats['Dominant planes detected'] = f"{planes_detected}/{len(feature_database)}"
    
    # Analyze density
    densities = [features.get('local_density', 0) for features in feature_database if features and 'local_density' in features]
    if densities:
        stats['Avg local density'] = f"{np.mean(densities):.3f} m"
    
    return stats

def save_non_repetitive_results(results, final_poses, combined_cloud=None, force_z_zero=False):
    """Save results from non-repetitive processing"""
    import os
    
    # Create output directory
    output_dir = "./output/non_repetitive_lidar"
    if force_z_zero:
        output_dir += "_z_zero"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save poses to CSV
    poses_file = os.path.join(output_dir, "final_poses.csv")
    pose_data = []
    
    for i, (result, pose) in enumerate(zip(results, final_poses)):
        if pose is not None:
            pose_data.append({
                'scan_index': i,
                'scan_file': result.get('scan_file', f'scan_{i}'),
                'x': pose[0],
                'y': pose[1], 
                'z': pose[2],
                'yaw_rad': pose[3],
                'yaw_deg': np.degrees(pose[3]),
                'prediction_confidence': result.get('prediction_confidence', 0.0),
                'z_redistributed': result.get('z_redistributed', False)
            })
    
    if pose_data:
        df = pd.DataFrame(pose_data)
        df.to_csv(poses_file, index=False)
        print(f"Saved poses to: {poses_file}")
    
    # Save combined cloud
    if combined_cloud is not None:
        cloud_filename = "combined_map_z_zero.pcd" if force_z_zero else "combined_map.pcd"
        cloud_file = os.path.join(output_dir, cloud_filename)
        success = o3d.io.write_point_cloud(cloud_file, combined_cloud)
        if success:
            print(f"Saved combined cloud to: {cloud_file}")
    
    # Save detailed results
    results_file = os.path.join(output_dir, "processing_results.json")
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = []
    for result in results:
        json_result = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                json_result[key] = value.tolist()
            elif key == 'features':
                # Simplified features for JSON
                json_result[key] = 'extracted' if value else 'failed'
            else:
                json_result[key] = value
        json_results.append(json_result)
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved results to: {results_file}")

# Example usage function
def main_non_repetitive_lidar():
    """Example usage for non-repetitive LiDAR processing"""
    
    # Configure paths
    observation_folder = "./testdata/Charm"  # Replace with your path
    
    print("=== Non-Repetitive LiDAR Scan Processing ===")
    print("Features:")
    print("- Adaptive feature-based prediction")
    print("- No velocity assumptions")
    print("- Geometric consistency checking")
    print("- Automatic strategy adaptation")
    print("- Handles unpredictable motion patterns")
    print("- Optional Z=0 mode with redistribution")
    print("////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////")
    
    try:
        # Process the scans with Z=0 mode enabled
        current_time = time.time()
        results, pred_poses, obs_poses, final_poses = process_non_repetitive_lidar_scans(
            observation_folder,
            visualize=True,
            observation_step_size=1,  # Process every 5th scan for efficiency
            observation_start_index=0,
            max_observation_clouds=250,  # Limit for testing
            force_z_zero=True,  # Enable Z=0 mode
            z_redistribution_method='prediction'  # Use prediction-based redistribution
        )
        elapsed_time = time.time() - current_time 
        print (elapsed_time/50)
        print(f"\n=== Processing Complete ===")
        successful_scans = len([r for r in results if 'error' not in r])
        print(f"Successfully processed: {successful_scans}/{len(results)} scans")
    
        if final_poses and any(pose is not None for pose in final_poses):
            # Create combined cloud
            from Pctools import load_pcd_files
            observation_pcds = load_pcd_files(observation_folder, 5, 0, 250)
            combined_cloud = create_combined_cloud(observation_pcds, final_poses)
            
            # Save results
            save_non_repetitive_results(results, final_poses, combined_cloud, force_z_zero=True)
            
            print(f"\nKey achievements:")
            print(f"- Processed non-repetitive LiDAR scans successfully")
            print(f"- Adaptive prediction without velocity assumptions")
            print(f"- Z-coordinate forced to 0 with intelligent redistribution")
            print(f"- Generated combined 2D map")
            print(f"- Results saved to ./output/non_repetitive_lidar_z_zero/")

        return results, pred_poses, obs_poses, final_poses
        
    except Exception as e:
        print(f"Error in processing: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

if __name__ == "__main__":
    main_non_repetitive_lidar()