#!/usr/bin/env python3
"""
ROS2 Non-Repetitive LiDAR Processor (PointCloud2 -> Open3D) + Live Open3D visualization
WITH ALL PARAMETERS LOADED FROM A YAML CONFIG FILE (ROS2 params).

MODIFIED: Publishes
  - LiDAR odometry (nav_msgs/Odometry)
  - Map pointcloud (sensor_msgs/PointCloud2) INCLUDING INTENSITY (x,y,z,intensity)
  - Visualizes intensity as grayscale colors in Open3D (latest + map)
  - (Optional) TF map->odom and odom->base_link

Run:
  ros2 run <your_pkg> ros_non_rep.py --ros-args --params-file config/non_rep_lidar.yaml

Example YAML:
  non_rep_lidar:
    ros__parameters:
      lidar_topic: "/points"
      queue_size: 10
      step_decimation: 5
      max_scans: -1
      accumulate_between_decimation: true
      accumulate_voxel: 0.10
      accumulate_max_points: 1500000
      force_z_zero: true
      z_redistribution_method: "prediction"
      fixed_weights: false
      feature_weight: 0.3
      geometric_weight: 0.4
      temporal_weight: 0.3
      freeze_adaptation: false
      visualize: true
      map_voxel: 0.15
      use_pctools_gicp: true
      gicp_max_corr_distance: 2.0
      gicp_voxel_size: 0.2
      gicp_max_iterations: 50

      publish_odom: true
      odom_topic: "/lidar/odom"
      publish_map: true
      map_topic: "/lidar/map"
      publish_tf: false
      map_frame: "map"
      odom_frame: "odom"
      base_frame: "base_link"
      map_publish_voxel: 0.15
      map_publish_max_points: 800000
      map_publish_every_n_scans: 1
"""

import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

# ROS2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2

from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import tf2_ros


# =========================
# Data model
# =========================
@dataclass
class ScanState:
    pose: np.ndarray  # [x, y, z, yaw]
    uncertainty: np.ndarray  # 4x4 covariance matrix
    confidence: float  # confidence score
    scan_features: Dict  # extracted features


# =========================
# Processor (your algorithm; kept as-is logically)
# =========================
class NonRepetitiveLiDARProcessor:
    def __init__(self,
                 adaptive_threshold: float = 0.9,
                 feature_weight: float = 0.3,
                 geometric_weight: float = 0.4,
                 temporal_weight: float = 0.3,
                 force_z_zero: bool = False,
                 z_redistribution_method: str = 'prediction'):
        self.adaptive_threshold = adaptive_threshold
        self.feature_weight = feature_weight
        self.geometric_weight = geometric_weight
        self.temporal_weight = temporal_weight
        self.force_z_zero = force_z_zero
        self.z_redistribution_method = z_redistribution_method

        self.scan_states: List[ScanState] = []
        self.feature_database: List[Dict] = []
        self.motion_patterns: List[str] = []

        # feature extraction params
        self.voxel_size = 0.1
        self.normal_radius = 0.5
        self.fpfh_radius = 1.0

    def redistribute_z_component(self, pose: np.ndarray, predicted_pose: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.force_z_zero or abs(pose[2]) < 1e-6:
            return pose.copy()
        out = pose.copy()
        out[2] = 0.0
        return out

    def extract_scan_features(self, cloud: o3d.geometry.PointCloud) -> Dict:
        features: Dict = {}
        try:
            points = np.asarray(cloud.points)
            if len(points) == 0:
                return features

            # 1) stats
            features['point_count'] = int(len(points))
            features['centroid'] = np.mean(points, axis=0)
            features['std_dev'] = np.std(points, axis=0)
            features['bounding_box'] = {
                'min': np.min(points, axis=0),
                'max': np.max(points, axis=0),
                'extent': np.max(points, axis=0) - np.min(points, axis=0)
            }

            # 2) downsample
            cloud_ds = cloud.voxel_down_sample(self.voxel_size) if len(points) > 1000 else cloud

            # 3) normals
            if len(cloud_ds.points) > 10:
                cloud_ds.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_radius, max_nn=30)
                )
                normals = np.asarray(cloud_ds.normals)
                if len(normals) > 0:
                    features['normal_distribution'] = {
                        'mean': np.mean(normals, axis=0),
                        'std': np.std(normals, axis=0)
                    }

            # 4) FPFH
            if len(cloud_ds.points) > 50 and cloud_ds.has_normals():
                fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    cloud_ds,
                    o3d.geometry.KDTreeSearchParamHybrid(radius=self.fpfh_radius, max_nn=100)
                )
                features['fpfh_histogram'] = np.asarray(fpfh.data).mean(axis=1)

            # 5) plane
            if len(cloud_ds.points) > 100:
                plane_model, inliers = cloud_ds.segment_plane(
                    distance_threshold=0.1, ransac_n=3, num_iterations=1000
                )
                if len(inliers) > 50:
                    features['dominant_plane'] = {
                        'normal': plane_model[:3],
                        'distance': plane_model[3],
                        'inlier_ratio': float(len(inliers) / len(cloud_ds.points))
                    }

            # 6) height profile
            z = points[:, 2]
            features['height_profile'] = {
                'min_height': float(np.min(z)),
                'max_height': float(np.max(z)),
                'mean_height': float(np.mean(z)),
                'height_variance': float(np.var(z))
            }

            # 7) density
            if len(points) > 100:
                sample_idx = np.random.choice(len(points), min(100, len(points)), replace=False)
                sample_pts = points[sample_idx]
                distances = cdist(sample_pts, points)
                k_nearest = np.sort(distances, axis=1)[:, 1:6]
                features['local_density'] = float(np.mean(k_nearest))

            # 8) PCA
            if len(points) > 20:
                pca = PCA(n_components=3)
                pca.fit(points)
                features['shape_complexity'] = {
                    'explained_variance_ratio': pca.explained_variance_ratio_,
                    'linearity': float(pca.explained_variance_ratio_[0]),
                    'planarity': float(pca.explained_variance_ratio_[1]),
                    'sphericity': float(pca.explained_variance_ratio_[2]),
                }

        except Exception as e:
            print(f"Error extracting features: {e}")
            features['extraction_error'] = str(e)

        return features

    def compute_feature_similarity(self, f1: Dict, f2: Dict) -> float:
        if not f1 or not f2:
            return 0.0
        sims = []
        try:
            if 'point_count' in f1 and 'point_count' in f2:
                a, b = float(f1['point_count']), float(f2['point_count'])
                sims.append(min(a, b) / max(a, b, 1e-12))

            if 'centroid' in f1 and 'centroid' in f2:
                if self.force_z_zero:
                    d = float(np.linalg.norm(f1['centroid'][:2] - f2['centroid'][:2]))
                else:
                    d = float(np.linalg.norm(f1['centroid'] - f2['centroid']))
                sims.append(max(0.0, 1.0 - d / 50.0))

            if 'bounding_box' in f1 and 'bounding_box' in f2:
                bb1, bb2 = f1['bounding_box'], f2['bounding_box']
                e1 = bb1['extent'][:2] if self.force_z_zero else bb1['extent']
                e2 = bb2['extent'][:2] if self.force_z_zero else bb2['extent']
                denom = float(np.prod(np.maximum(e1, e2)))
                if denom > 0:
                    sims.append(float(np.prod(np.minimum(e1, e2)) / denom))

            if 'fpfh_histogram' in f1 and 'fpfh_histogram' in f2:
                h1, h2 = f1['fpfh_histogram'], f2['fpfh_histogram']
                if len(h1) == len(h2):
                    dot = float(np.dot(h1, h2))
                    norm = float(np.linalg.norm(h1) * np.linalg.norm(h2))
                    if norm > 1e-12:
                        sims.append(max(0.0, dot / norm))

            if 'height_profile' in f1 and 'height_profile' in f2:
                hp1, hp2 = f1['height_profile'], f2['height_profile']
                if not self.force_z_zero:
                    r1 = float(hp1['max_height'] - hp1['min_height'])
                    r2 = float(hp2['max_height'] - hp2['min_height'])
                    sims.append(min(r1, r2) / max(r1, r2, 1e-12))
                else:
                    v1 = float(hp1['height_variance'])
                    v2 = float(hp2['height_variance'])
                    sims.append(min(v1, v2) / max(v1, v2, 1e-12))

            if 'local_density' in f1 and 'local_density' in f2:
                d1, d2 = float(f1['local_density']), float(f2['local_density'])
                sims.append(min(d1, d2) / max(d1, d2, 1e-12))

        except Exception as e:
            print(f"Error computing similarity: {e}")

        return float(np.mean(sims)) if sims else 0.0

    def predict_pose_feature_based(self, current_features: Dict) -> Tuple[Optional[np.ndarray], float]:
        if len(self.feature_database) < 2:
            return None, 0.0

        sims = []
        for i, (feat, st) in enumerate(zip(self.feature_database, self.scan_states)):
            sims.append((self.compute_feature_similarity(current_features, feat), i, st))
        sims.sort(key=lambda x: x[0], reverse=True)

        if len(sims) < 2:
            return None, 0.0

        best_sim, _, best_state = sims[0]
        sec_sim, _, sec_state = sims[1]
        if best_sim < 0.3:
            return None, 0.0

        w1 = best_sim / (best_sim + sec_sim + 1e-12)
        w2 = sec_sim / (best_sim + sec_sim + 1e-12)
        pred = w1 * best_state.pose + w2 * sec_state.pose
        conf = float((best_sim + sec_sim) / 2.0)
        return pred, conf

    def predict_pose_geometric_consistency(self) -> Tuple[Optional[np.ndarray], float]:
        if len(self.scan_states) < 3:
            return None, 0.0

        recent = self.scan_states[-min(5, len(self.scan_states)):]
        poses = [s.pose for s in recent]
        if len(poses) < 3:
            return None, 0.0

        pred = np.zeros(4, dtype=float)
        confs = []
        dims = [0, 1, 3] if self.force_z_zero else [0, 1, 2, 3]

        for dim in dims:
            vals = np.array([p[dim] for p in poses], dtype=float)
            x = np.arange(len(vals), dtype=float)
            deg = min(2, len(vals) - 1)
            coeffs = np.polyfit(x, vals, deg)
            pred[dim] = float(np.polyval(coeffs, float(len(vals))))
            fitted = np.polyval(coeffs, x)
            mse = float(np.mean((vals - fitted) ** 2))
            confs.append(max(0.0, 1.0 - mse))

        if self.force_z_zero:
            pred[2] = 0.0

        pred[3] = float(np.arctan2(np.sin(pred[3]), np.cos(pred[3])))
        return pred, float(np.mean(confs)) if confs else 0.0

    def predict_pose_adaptive(self, current_features: Dict) -> Tuple[Optional[np.ndarray], float]:
        preds = []

        fp, fc = self.predict_pose_feature_based(current_features)
        if fp is not None:
            preds.append((fp, fc, "feature"))

        gp, gc = self.predict_pose_geometric_consistency()
        if gp is not None:
            preds.append((gp, gc, "geometric"))

        if len(self.scan_states) >= 2:
            last = self.scan_states[-1].pose
            prev = self.scan_states[-2].pose
            extrap = last + 0.3 * (last - prev)
            extrap[3] = np.arctan2(np.sin(extrap[3]), np.cos(extrap[3]))
            if self.force_z_zero:
                extrap[2] = 0.0
            preds.append((extrap, 0.4, "extrapolation"))

        if not preds:
            return None, 0.0
        if len(preds) == 1:
            return preds[0][0], float(preds[0][1])

        total_w = 0.0
        weighted = np.zeros(4, dtype=float)
        for pose, conf, strat in preds:
            strat_w = self.feature_weight if strat == "feature" else (self.geometric_weight if strat == "geometric" else self.temporal_weight)
            w = float(conf) * float(strat_w)
            weighted += w * pose
            total_w += w

        if total_w > 1e-12:
            out = weighted / total_w
            if self.force_z_zero:
                out[2] = 0.0
            return out, float(total_w / len(preds))

        return None, 0.0

    def update_with_observation(self,
                                observed_pose: np.ndarray,
                                scan_features: Dict,
                                registration_confidence: float = 1.0,
                                predicted_pose: Optional[np.ndarray] = None):
        final_pose = self.redistribute_z_component(observed_pose, predicted_pose)

        base_unc = 0.1
        unc = np.eye(4) * (base_unc / max(registration_confidence, 1e-6)) ** 2

        self.scan_states.append(ScanState(
            pose=final_pose.copy(),
            uncertainty=unc,
            confidence=float(registration_confidence),
            scan_features=scan_features
        ))
        self.feature_database.append(scan_features)

        max_hist = 20
        if len(self.scan_states) > max_hist:
            self.scan_states.pop(0)
            self.feature_database.pop(0)

        self._analyze_motion_patterns()

    def _analyze_motion_patterns(self):
        if len(self.scan_states) < 5:
            return
        recent = [s.pose for s in self.scan_states[-5:]]
        moves = []
        for i in range(1, len(recent)):
            if self.force_z_zero:
                moves.append(float(np.linalg.norm(recent[i][:2] - recent[i - 1][:2])))
            else:
                moves.append(float(np.linalg.norm(recent[i][:3] - recent[i - 1][:3])))

        avg = float(np.mean(moves))
        std = float(np.std(moves))

        if avg < 0.1:
            pattern = "stationary"
        elif std / (avg + 1e-6) < 0.3:
            pattern = "smooth"
        elif std / (avg + 1e-6) > 1.0:
            pattern = "erratic"
        else:
            pattern = "variable"

        self.motion_patterns.append(pattern)
        if len(self.motion_patterns) > 10:
            self.motion_patterns.pop(0)

        if pattern == "erratic":
            self.feature_weight, self.geometric_weight, self.temporal_weight = 0.5, 0.2, 0.3
        elif pattern == "smooth":
            self.feature_weight, self.geometric_weight, self.temporal_weight = 0.2, 0.5, 0.3
        else:
            self.feature_weight, self.geometric_weight, self.temporal_weight = 0.35, 0.35, 0.3

    def get_current_state(self) -> Optional[ScanState]:
        return self.scan_states[-1] if self.scan_states else None


# =========================
# Helpers
# =========================
def transformation_to_pose(T: np.ndarray) -> np.ndarray:
    t = T[:3, 3]
    R = T[:3, :3]
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([t[0], t[1], t[2], yaw], dtype=float)


def yaw_to_quat(yaw: float):
    half = 0.5 * float(yaw)
    return 0.0, 0.0, float(np.sin(half)), float(np.cos(half))


def estimate_registration_confidence(cloud1, cloud2, transformation, sample_size=100):
    try:
        pts1 = np.asarray(cloud1.points)
        pts2 = np.asarray(cloud2.points)
        if len(pts1) == 0 or len(pts2) == 0:
            return 0.1

        idx = np.random.choice(len(pts1), min(sample_size, len(pts1)), replace=False)
        sp = pts1[idx]
        sp_h = np.column_stack([sp, np.ones(len(sp))])
        tp = (transformation @ sp_h.T).T[:, :3]

        from scipy.spatial import cKDTree
        tree = cKDTree(pts2)
        d, _ = tree.query(tp)
        avg_d = float(np.mean(d))
        return float(max(0.1, min(1.0, 1.0 - avg_d / 2.0)))
    except Exception as e:
        print(f"Error estimating confidence: {e}")
        return 0.5


def _normalize_intensity(intensity: np.ndarray) -> np.ndarray:
    """
    Convert arbitrary intensity scale into [0,1] float for Open3D colors and publishing.
    - If already mostly in [0,1], keep.
    - Otherwise scale by max (robust).
    """
    if intensity.size == 0:
        return intensity.astype(np.float32, copy=False)

    inten = intensity.astype(np.float32, copy=False)
    m = float(np.nanmax(inten)) if np.isfinite(inten).any() else 0.0
    if m <= 0.0:
        return np.zeros_like(inten, dtype=np.float32)

    # Heuristic: if max is <= ~1.5 we assume already normalized
    if m <= 1.5:
        out = np.clip(inten, 0.0, 1.0)
        return out.astype(np.float32, copy=False)

    # Otherwise scale by max to [0,1]
    out = np.clip(inten / m, 0.0, 1.0)
    return out.astype(np.float32, copy=False)


# =========================
# ROS2 <-> Open3D conversion (NOW WITH INTENSITY)
# =========================
def pointcloud2_to_xyz_i(msg: PointCloud2) -> Tuple[np.ndarray, np.ndarray]:
    field_names = [f.name for f in msg.fields]
    has_intensity = "intensity" in field_names

    pts = []
    intens = []
    if has_intensity:
        for p in pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
            pts.append([p[0], p[1], p[2]])
            intens.append(p[3])
    else:
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            pts.append([p[0], p[1], p[2]])
        intens = [0.0] * len(pts)

    if not pts:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    xyz = np.asarray(pts, dtype=np.float32)
    intensity = np.asarray(intens, dtype=np.float32)
    return xyz, intensity


def xyzi_to_open3d_cloud(xyz: np.ndarray, intensity: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    if xyz.size == 0:
        return cloud

    cloud.points = o3d.utility.Vector3dVector(xyz.astype(np.float64, copy=False))

    # Visualize intensity as grayscale colors
    inten01 = _normalize_intensity(intensity)
    if inten01.size == xyz.shape[0]:
        colors = np.stack([inten01, inten01, inten01], axis=1).astype(np.float64, copy=False)
        cloud.colors = o3d.utility.Vector3dVector(colors)

    return cloud


def open3d_cloud_to_pointcloud2_xyzi(cloud: o3d.geometry.PointCloud, header: Header) -> PointCloud2:
    pts = np.asarray(cloud.points)
    if pts.size == 0:
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        return pc2.create_cloud(header, fields, [])

    if cloud.has_colors():
        cols = np.asarray(cloud.colors)
        intensity = cols[:, 0].astype(np.float32, copy=False)  # grayscale encoding
    else:
        intensity = np.zeros((pts.shape[0],), dtype=np.float32)

    pts32 = pts.astype(np.float32, copy=False)
    intensity = intensity.reshape(-1).astype(np.float32, copy=False)

    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    data = [(float(p[0]), float(p[1]), float(p[2]), float(i)) for p, i in zip(pts32, intensity)]
    return pc2.create_cloud(header, fields, data)


# =========================
# Live Open3D visualization
# =========================
class LiveOpen3D:
    def __init__(self, window_name="ROS2 Non-Repetitive LiDAR", width=1400, height=900):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name, width=width, height=height)

        self.latest = o3d.geometry.PointCloud()
        self.map = o3d.geometry.PointCloud()
        self._latest_added = False
        self._map_added = False

    def update(self, latest_cloud: Optional[o3d.geometry.PointCloud], map_cloud: Optional[o3d.geometry.PointCloud]):
        if latest_cloud is not None:
            self.latest = latest_cloud
            if not self._latest_added:
                self.vis.add_geometry(self.latest)
                self._latest_added = True
            else:
                self.vis.update_geometry(self.latest)

        if map_cloud is not None:
            self.map = map_cloud
            if not self._map_added:
                self.vis.add_geometry(self.map)
                self._map_added = True
            else:
                self.vis.update_geometry(self.map)

        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()


# =========================
# GICP
# =========================
def apply_gicp_open3d(source: o3d.geometry.PointCloud,
                      target: o3d.geometry.PointCloud,
                      voxel_size: float = 0.2,
                      max_corr_distance: float = 2.0,
                      max_iterations: int = 50) -> np.ndarray:
    """Open3D GICP fallback. Returns 4x4 transform (source -> target)."""
    if len(source.points) < 30 or len(target.points) < 30:
        return np.eye(4, dtype=float)

    src = source.voxel_down_sample(float(voxel_size)) if voxel_size > 0 else source
    tgt = target.voxel_down_sample(float(voxel_size)) if voxel_size > 0 else target

    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

    res = o3d.pipelines.registration.registration_generalized_icp(
        src, tgt,
        max_correspondence_distance=float(max_corr_distance),
        init=np.eye(4, dtype=float),
        estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iterations))
    )
    return res.transformation


# =========================
# ROS2 Node
# =========================
class NonRepetitiveLiDARRos2Node(Node):
    def __init__(self):
        super().__init__("non_rep_lidar")

        def p(name, default):
            self.declare_parameter(name, default)
            return self.get_parameter(name).value

        # ---- ROS/Input params
        self.lidar_topic = str(p("lidar_topic", "/points"))
        self.queue_size = int(p("queue_size", 10))

        # ---- Publishing params
        self.publish_odom = bool(p("publish_odom", True))
        self.odom_topic = str(p("odom_topic", "/lidar/odom"))
        self.publish_map = bool(p("publish_map", True))
        self.map_topic = str(p("map_topic", "/lidar/map"))

        self.publish_tf = bool(p("publish_tf", False))
        self.map_frame = str(p("map_frame", "map"))
        self.odom_frame = str(p("odom_frame", "odom"))
        self.base_frame = str(p("base_frame", "base_link"))

        self.map_publish_voxel = float(p("map_publish_voxel", 0.15))
        self.map_publish_max_points = int(p("map_publish_max_points", 800_000))
        self.map_publish_every_n_scans = max(1, int(p("map_publish_every_n_scans", 1)))

        # ---- Processing control
        self.step_decimation = max(1, int(p("step_decimation", 1)))
        max_scans = int(p("max_scans", -1))
        self.max_scans: Optional[int] = None if max_scans < 0 else max_scans

        # ---- Accumulation between decimation ticks
        self.accumulate_between_decimation = bool(p("accumulate_between_decimation", False))
        acc_vox = float(p("accumulate_voxel", 0.1))
        self.accumulate_voxel: Optional[float] = None if acc_vox < 0 else acc_vox
        self.accumulate_max_points = int(p("accumulate_max_points", 1_500_000))

        # ---- Z handling
        self.force_z_zero = bool(p("force_z_zero", False))
        self.z_redistribution_method = str(p("z_redistribution_method", "prediction"))

        # ---- Prediction weights / adaptation
        self.fixed_weights = bool(p("fixed_weights", False))
        self.feature_weight = float(p("feature_weight", 0.3))
        self.geometric_weight = float(p("geometric_weight", 0.4))
        self.temporal_weight = float(p("temporal_weight", 0.3))
        self.freeze_adaptation = bool(p("freeze_adaptation", False))

        # ---- Visualization
        self.visualize = bool(p("visualize", True))
        self.map_voxel = float(p("map_voxel", 0.15))

        # ---- GICP selection & parameters
        self.use_pctools_gicp = bool(p("use_pctools_gicp", True))
        self.gicp_max_corr_distance = float(p("gicp_max_corr_distance", 2.0))
        self.gicp_voxel_size = float(p("gicp_voxel_size", 0.2))
        self.gicp_max_iterations = int(p("gicp_max_iterations", 50))

        # ---- Processor
        self.processor = NonRepetitiveLiDARProcessor(
            force_z_zero=self.force_z_zero,
            z_redistribution_method=self.z_redistribution_method
        )
        if self.fixed_weights:
            self.processor.feature_weight = self.feature_weight
            self.processor.geometric_weight = self.geometric_weight
            self.processor.temporal_weight = self.temporal_weight
        if self.freeze_adaptation:
            self.processor._analyze_motion_patterns = lambda *a, **k: None

        self.apply_gicp_func = self._resolve_gicp()

        # ---- Runtime state
        self.prev_cloud: Optional[o3d.geometry.PointCloud] = None
        self.cumulative_transform: Optional[np.ndarray] = None

        self.map_cloud = o3d.geometry.PointCloud()
        self._buffer_cloud = o3d.geometry.PointCloud()

        self.msg_counter = 0
        self.scan_counter = 0

        self.viewer = LiveOpen3D(
            window_name="Non-Repetitive LiDAR Map (Intensity, Z=0)" if self.force_z_zero else "Non-Repetitive LiDAR Map (Intensity)"
        ) if self.visualize else None

        # ---- Publishers
        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 10) if self.publish_odom else None
        self.map_pub = self.create_publisher(PointCloud2, self.map_topic, 1) if self.publish_map else None
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self) if self.publish_tf else None

        # Subscriber
        self.sub = self.create_subscription(PointCloud2, self.lidar_topic, self.cb_cloud, self.queue_size)

        self.get_logger().info("=== Non-Rep LiDAR ROS2 Node (Intensity Map) ===")
        self.get_logger().info(f"topic={self.lidar_topic} queue={self.queue_size}")
        self.get_logger().info(f"publish_map={self.publish_map} map_topic={self.map_topic} (x,y,z,intensity)")

    def _resolve_gicp(self):
        if self.use_pctools_gicp:
            try:
                from Pctools import apply_gicp_direct
                self.get_logger().info("Using Pctools.apply_gicp_direct")
                return apply_gicp_direct
            except Exception as e:
                self.get_logger().warn(f"Pctools import failed: {e}. Using Open3D GICP fallback.")

        def _gicp(src, tgt):
            return apply_gicp_open3d(
                src, tgt,
                voxel_size=self.gicp_voxel_size,
                max_corr_distance=self.gicp_max_corr_distance,
                max_iterations=self.gicp_max_iterations
            )
        return _gicp

    def _flush_or_buffer(self, cloud_raw: o3d.geometry.PointCloud) -> Optional[o3d.geometry.PointCloud]:
        if not self.accumulate_between_decimation:
            if (self.msg_counter - 1) % self.step_decimation != 0:
                return None
            return cloud_raw

        if len(cloud_raw.points) > 0:
            self._buffer_cloud += cloud_raw
            if len(self._buffer_cloud.points) > self.accumulate_max_points:
                vx = self.accumulate_voxel if self.accumulate_voxel is not None else 0.1
                self._buffer_cloud = self._buffer_cloud.voxel_down_sample(float(vx))

        if (self.msg_counter - 1) % self.step_decimation != 0:
            return None

        merged = self._buffer_cloud
        if self.accumulate_voxel is not None and len(merged.points) > 0:
            merged = merged.voxel_down_sample(float(self.accumulate_voxel))

        self._buffer_cloud = o3d.geometry.PointCloud()
        return merged

    def _publish_odom_and_tf(self, stamp, final_pose: np.ndarray, st: Optional[ScanState]):
        if not self.publish_odom and not self.publish_tf:
            return

        x, y, z, yaw = [float(v) for v in final_pose]
        qx, qy, qz, qw = yaw_to_quat(yaw)

        if self.odom_pub is not None:
            odom = Odometry()
            odom.header.stamp = stamp
            odom.header.frame_id = self.odom_frame
            odom.child_frame_id = self.base_frame

            odom.pose.pose.position.x = x
            odom.pose.pose.position.y = y
            odom.pose.pose.position.z = z
            odom.pose.pose.orientation.x = qx
            odom.pose.pose.orientation.y = qy
            odom.pose.pose.orientation.z = qz
            odom.pose.pose.orientation.w = qw

            cov6 = np.zeros((6, 6), dtype=float)
            if st is not None and isinstance(st.uncertainty, np.ndarray) and st.uncertainty.shape == (4, 4):
                cov6[0, 0] = float(st.uncertainty[0, 0])
                cov6[1, 1] = float(st.uncertainty[1, 1])
                cov6[2, 2] = float(st.uncertainty[2, 2])
                cov6[5, 5] = float(st.uncertainty[3, 3])
            odom.pose.covariance = cov6.reshape(-1).tolist()

            self.odom_pub.publish(odom)

        if self.tf_broadcaster is not None:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.base_frame
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = z
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(t)

            t2 = TransformStamped()
            t2.header.stamp = stamp
            t2.header.frame_id = self.map_frame
            t2.child_frame_id = self.odom_frame
            t2.transform.translation.x = 0.0
            t2.transform.translation.y = 0.0
            t2.transform.translation.z = 0.0
            t2.transform.rotation.x = 0.0
            t2.transform.rotation.y = 0.0
            t2.transform.rotation.z = 0.0
            t2.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(t2)

    def _publish_map_cloud(self, stamp):
        if self.map_pub is None:
            return
        if self.scan_counter % self.map_publish_every_n_scans != 0:
            return

        cloud_to_pub = self.map_cloud

        if len(cloud_to_pub.points) > 0 and self.map_publish_voxel > 0:
            cloud_to_pub = cloud_to_pub.voxel_down_sample(float(self.map_publish_voxel))

        if self.map_publish_max_points > 0 and len(cloud_to_pub.points) > self.map_publish_max_points:
            pts = np.asarray(cloud_to_pub.points)
            cols = np.asarray(cloud_to_pub.colors) if cloud_to_pub.has_colors() else None

            idx = np.random.choice(len(pts), self.map_publish_max_points, replace=False)
            pts = pts[idx]
            tmp = o3d.geometry.PointCloud()
            tmp.points = o3d.utility.Vector3dVector(pts.astype(np.float64, copy=False))

            if cols is not None and len(cols) == len(np.asarray(cloud_to_pub.points)):
                cols = cols[idx]
                tmp.colors = o3d.utility.Vector3dVector(cols.astype(np.float64, copy=False))

            cloud_to_pub = tmp

        header = Header()
        header.stamp = stamp
        header.frame_id = self.map_frame
        self.map_pub.publish(open3d_cloud_to_pointcloud2_xyzi(cloud_to_pub, header))

    def cb_cloud(self, msg: PointCloud2):
        self.msg_counter += 1

        xyz, intensity = pointcloud2_to_xyz_i(msg)
        cloud_raw = xyzi_to_open3d_cloud(xyz, intensity)

        cloud = self._flush_or_buffer(cloud_raw)
        if cloud is None or len(cloud.points) == 0:
            return

        if self.max_scans is not None and self.scan_counter >= self.max_scans:
            return

        stamp = msg.header.stamp

        try:
            feat = self.processor.extract_scan_features(cloud)
            pred_pose, _ = self.processor.predict_pose_adaptive(feat)

            if self.prev_cloud is not None and len(self.prev_cloud.points) > 0:
                T = self.apply_gicp_func(self.prev_cloud, cloud)

                self.cumulative_transform = T if self.cumulative_transform is None else (self.cumulative_transform @ T)
                observed_pose = transformation_to_pose(self.cumulative_transform)

                reg_conf = estimate_registration_confidence(self.prev_cloud, cloud, T)
                self.processor.update_with_observation(observed_pose, feat, reg_conf, pred_pose)

                st = self.processor.get_current_state()
                final_pose = st.pose.copy() if st is not None else observed_pose.copy()
            else:
                if self.cumulative_transform is None:
                    self.cumulative_transform = np.eye(4, dtype=float)
                final_pose = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
                if self.force_z_zero:
                    final_pose[2] = 0.0
                self.processor.update_with_observation(final_pose, feat, registration_confidence=0.3, predicted_pose=pred_pose)
                st = self.processor.get_current_state()

            # Publish odom (+tf)
            self._publish_odom_and_tf(stamp=stamp, final_pose=final_pose, st=st)

            # Accumulate intensity-colored map
            Tmap = np.eye(4, dtype=float)
            yaw = float(final_pose[3])
            Tmap[0, 0] = np.cos(yaw);  Tmap[0, 1] = -np.sin(yaw)
            Tmap[1, 0] = np.sin(yaw);  Tmap[1, 1] =  np.cos(yaw)
            Tmap[0, 3] = float(final_pose[0])
            Tmap[1, 3] = float(final_pose[1])
            Tmap[2, 3] = float(final_pose[2])

            cur_in_map = o3d.geometry.PointCloud(cloud)
            cur_in_map.transform(Tmap)
            self.map_cloud += cur_in_map

            if self.map_voxel > 0 and len(self.map_cloud.points) > 2_000_000:
                self.map_cloud = self.map_cloud.voxel_down_sample(float(self.map_voxel))

            # Publish map with intensity field
            self._publish_map_cloud(stamp=stamp)

            if self.viewer is not None:
                self.viewer.update(latest_cloud=cloud, map_cloud=self.map_cloud)

        except Exception as e:
            self.get_logger().error(f"Error processing scan {self.scan_counter}: {e}")

        self.prev_cloud = cloud
        self.scan_counter += 1

        if self.max_scans is not None and self.scan_counter >= self.max_scans:
            self.get_logger().info("Max scans reached. Shutting down.")
            rclpy.shutdown()

    def shutdown(self):
        if self.viewer is not None:
            self.viewer.close()


def main():
    rclpy.init()
    node = NonRepetitiveLiDARRos2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()