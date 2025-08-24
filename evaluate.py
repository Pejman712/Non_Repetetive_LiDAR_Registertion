import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
import json

def find_closest_points(traj_df, gt_df):
    """Find closest points between trajectory and ground truth based on Euclidean distance"""
    traj_points = traj_df[['x', 'y']].values
    gt_points = gt_df[['x', 'y']].values
    
    distances = cdist(traj_points, gt_points)
    closest_gt_indices = np.argmin(distances, axis=1)
    closest_distances = np.min(distances, axis=1)
    
    return closest_gt_indices, closest_distances

def match_by_closest_distance(traj_df, gt_df):
    """Match trajectory points to GT points based on closest distance"""
    closest_indices, distances = find_closest_points(traj_df, gt_df)
    
    matched_traj = traj_df.copy()
    matched_gt = gt_df.iloc[closest_indices].reset_index(drop=True)
    
    return matched_traj, matched_gt, distances

def match_by_same_points(traj_df, gt_df):
    """Match by ensuring same number of points through resampling"""
    min_length = min(len(traj_df), len(gt_df))
    
    traj_resampled = resample(traj_df, min_length)
    gt_resampled = resample(gt_df, min_length)
    
    return traj_resampled, gt_resampled

def compute_comprehensive_metrics(traj_df, gt_df, method='closest_distance'):
    """Compute comprehensive metrics and comparison matrices"""
    if method == 'closest_distance':
        matched_traj, matched_gt, distances = match_by_closest_distance(traj_df, gt_df)
        errors = distances
    elif method == 'same_points':
        matched_traj, matched_gt = match_by_same_points(traj_df, gt_df)
        errors = np.sqrt((matched_traj['x'] - matched_gt['x'])**2 + 
                        (matched_traj['y'] - matched_gt['y'])**2)
    else:
        raise ValueError("Method must be 'closest_distance' or 'same_points'")
    
    # Basic error metrics
    basic_metrics = {
        'mean_error': errors.mean(),
        'std_error': errors.std(),
        'rmse': np.sqrt(np.mean(errors ** 2)),
        'max_error': errors.max(),
        'min_error': errors.min(),
        'median_error': np.median(errors),
        'num_points': len(errors)
    }
    
    # Additional comparison metrics
    traj_points = matched_traj[['x', 'y']].values
    gt_points = matched_gt[['x', 'y']].values
    
    # 1. Hausdorff Distance (bidirectional)
    hausdorff_traj_to_gt = np.max(np.min(cdist(traj_points, gt_points), axis=1))
    hausdorff_gt_to_traj = np.max(np.min(cdist(gt_points, traj_points), axis=1))
    hausdorff_distance = max(hausdorff_traj_to_gt, hausdorff_gt_to_traj)
    
    # 2. Fréchet Distance (iterative implementation to avoid recursion issues)
    def frechet_distance_discrete(P, Q):
        # Limit the size to prevent memory/performance issues
        max_points = 1000
        if len(P) > max_points:
            indices = np.linspace(0, len(P)-1, max_points, dtype=int)
            P = P[indices]
        if len(Q) > max_points:
            indices = np.linspace(0, len(Q)-1, max_points, dtype=int)
            Q = Q[indices]
            
        ca = np.full((len(P), len(Q)), -1.0)
        
        # Fill the DP table iteratively
        for i in range(len(P)):
            for j in range(len(Q)):
                dist = np.linalg.norm(P[i] - Q[j])
                if i == 0 and j == 0:
                    ca[i, j] = dist
                elif i > 0 and j == 0:
                    ca[i, j] = max(ca[i-1, 0], dist)
                elif i == 0 and j > 0:
                    ca[i, j] = max(ca[0, j-1], dist)
                else:  # i > 0 and j > 0
                    ca[i, j] = max(min(ca[i-1, j], ca[i-1, j-1], ca[i, j-1]), dist)
        
        return ca[len(P)-1, len(Q)-1]
    
    try:
        frechet_dist = frechet_distance_discrete(traj_points, gt_points)
    except Exception as e:
        print(f"Warning: Fréchet distance calculation failed: {e}")
        frechet_dist = np.nan
    
    # 3. Path Length Comparison
    def path_length(points):
        if len(points) < 2:
            return 0
        return np.sum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    
    traj_length = path_length(traj_points)
    gt_length = path_length(gt_points)
    length_ratio = traj_length / gt_length if gt_length > 0 else float('inf')
    length_difference = abs(traj_length - gt_length)
    
    # 4. Area Between Curves (using trapezoidal rule)
    def area_between_curves(traj, gt):
        try:
            # Check if we have enough points
            if len(traj) < 2 or len(gt) < 2:
                return np.nan
                
            # Sort points by x-coordinate for interpolation
            traj_sorted = traj[np.argsort(traj[:, 0])]
            gt_sorted = gt[np.argsort(gt[:, 0])]
            
            # Check for duplicate x-values and handle them
            if len(np.unique(traj_sorted[:, 0])) < 2 or len(np.unique(gt_sorted[:, 0])) < 2:
                return np.nan
            
            # Create common x-axis
            x_min = max(np.min(traj_sorted[:, 0]), np.min(gt_sorted[:, 0]))
            x_max = min(np.max(traj_sorted[:, 0]), np.max(gt_sorted[:, 0]))
            
            if x_max <= x_min:
                return np.nan
                
            # Create interpolation points
            num_points = min(100, len(traj) + len(gt))  # Limit computation
            x_common = np.linspace(x_min, x_max, num_points)
            
            # Interpolate y-values
            traj_y_interp = np.interp(x_common, traj_sorted[:, 0], traj_sorted[:, 1])
            gt_y_interp = np.interp(x_common, gt_sorted[:, 0], gt_sorted[:, 1])
            
            # Calculate area between curves
            return np.trapz(np.abs(traj_y_interp - gt_y_interp), x_common)
            
        except Exception as e:
            print(f"Warning: Area between curves calculation failed: {e}")
            return np.nan
    
    area_between = area_between_curves(traj_points, gt_points)
    
    # 5. Angular Deviation
    def compute_angles(points):
        if len(points) < 3:
            return np.array([])
        vectors = np.diff(points, axis=0)
        # Handle zero vectors
        norms = np.linalg.norm(vectors, axis=1)
        valid_indices = norms > 1e-10
        if not np.any(valid_indices):
            return np.array([])
        vectors = vectors[valid_indices]
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        return angles
    
    try:
        traj_angles = compute_angles(traj_points)
        gt_angles = compute_angles(gt_points)
        
        if len(traj_angles) > 0 and len(gt_angles) > 0:
            # Match angles by interpolation to same length
            common_length = min(len(traj_angles), len(gt_angles))
            if common_length > 1:
                traj_angles_matched = np.interp(np.linspace(0, len(traj_angles)-1, common_length),
                                               np.arange(len(traj_angles)), traj_angles)
                gt_angles_matched = np.interp(np.linspace(0, len(gt_angles)-1, common_length),
                                             np.arange(len(gt_angles)), gt_angles)
                
                angle_diff = np.abs(np.angle(np.exp(1j * (traj_angles_matched - gt_angles_matched))))
                mean_angular_error = np.mean(angle_diff)
                max_angular_error = np.max(angle_diff)
            else:
                mean_angular_error = np.nan
                max_angular_error = np.nan
        else:
            mean_angular_error = np.nan
            max_angular_error = np.nan
    except Exception as e:
        print(f"Warning: Angular deviation calculation failed: {e}")
        mean_angular_error = np.nan
        max_angular_error = np.nan
    
    # 6. Correlation Metrics
    try:
        if len(matched_traj) == len(matched_gt) and len(matched_traj) > 1:
            x_correlation, _ = pearsonr(matched_traj['x'], matched_gt['x'])
            y_correlation, _ = pearsonr(matched_traj['y'], matched_gt['y'])
        else:
            x_correlation = np.nan
            y_correlation = np.nan
    except Exception as e:
        print(f"Warning: Correlation calculation failed: {e}")
        x_correlation = np.nan
        y_correlation = np.nan
    
    # 7. Centroid Distance
    traj_centroid = np.mean(traj_points, axis=0)
    gt_centroid = np.mean(gt_points, axis=0)
    centroid_distance = np.linalg.norm(traj_centroid - gt_centroid)
    
    # 8. Bounding Box Metrics
    traj_bbox = [np.min(traj_points[:, 0]), np.min(traj_points[:, 1]),
                 np.max(traj_points[:, 0]), np.max(traj_points[:, 1])]
    gt_bbox = [np.min(gt_points[:, 0]), np.min(gt_points[:, 1]),
               np.max(gt_points[:, 0]), np.max(gt_points[:, 1])]
    
    bbox_area_traj = (traj_bbox[2] - traj_bbox[0]) * (traj_bbox[3] - traj_bbox[1])
    bbox_area_gt = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    bbox_area_ratio = bbox_area_traj / bbox_area_gt if bbox_area_gt > 0 else float('inf')
    
    # 9. Overlap Metrics (Intersection over Union for bounding boxes)
    intersection_area = max(0, min(traj_bbox[2], gt_bbox[2]) - max(traj_bbox[0], gt_bbox[0])) * \
                       max(0, min(traj_bbox[3], gt_bbox[3]) - max(traj_bbox[1], gt_bbox[1]))
    union_area = bbox_area_traj + bbox_area_gt - intersection_area
    iou_bbox = intersection_area / union_area if union_area > 0 else 0
    
    # 10. Velocity Comparison (if applicable)
    def compute_velocities(points):
        if len(points) < 2:
            return np.array([])
        diff_points = np.diff(points, axis=0)
        velocities = np.sqrt(np.sum(diff_points**2, axis=1))
        return velocities
    
    try:
        traj_velocities = compute_velocities(traj_points)
        gt_velocities = compute_velocities(gt_points)
        
        if len(traj_velocities) > 0 and len(gt_velocities) > 0:
            # Match velocities by interpolation
            common_length = min(len(traj_velocities), len(gt_velocities))
            if common_length > 1:
                traj_vel_matched = np.interp(np.linspace(0, len(traj_velocities)-1, common_length),
                                            np.arange(len(traj_velocities)), traj_velocities)
                gt_vel_matched = np.interp(np.linspace(0, len(gt_velocities)-1, common_length),
                                          np.arange(len(gt_velocities)), gt_velocities)
                
                velocity_rmse = np.sqrt(np.mean((traj_vel_matched - gt_vel_matched)**2))
                
                # Check for constant velocities
                if np.std(traj_vel_matched) > 1e-10 and np.std(gt_vel_matched) > 1e-10:
                    velocity_correlation, _ = pearsonr(traj_vel_matched, gt_vel_matched)
                else:
                    velocity_correlation = np.nan
            else:
                velocity_rmse = np.nan
                velocity_correlation = np.nan
        else:
            velocity_rmse = np.nan
            velocity_correlation = np.nan
    except Exception as e:
        print(f"Warning: Velocity calculation failed: {e}")
        velocity_rmse = np.nan
        velocity_correlation = np.nan
    
    # Combine all metrics
    comprehensive_metrics = {
        **basic_metrics,
        'hausdorff_distance': hausdorff_distance,
        'frechet_distance': frechet_dist,
        'path_length_traj': traj_length,
        'path_length_gt': gt_length,
        'path_length_ratio': length_ratio,
        'path_length_diff': length_difference,
        'area_between_curves': area_between,
        'mean_angular_error': mean_angular_error,
        'max_angular_error': max_angular_error,
        'x_correlation': x_correlation,
        'y_correlation': y_correlation,
        'centroid_distance': centroid_distance,
        'bbox_area_ratio': bbox_area_ratio,
        'bbox_iou': iou_bbox,
        'velocity_rmse': velocity_rmse,
        'velocity_correlation': velocity_correlation
    }
    
    return comprehensive_metrics, matched_traj, matched_gt

def create_transformation_matrix(matched_traj, matched_gt):
    """Create transformation matrix to align trajectory with ground truth"""
    traj_centroid = matched_traj[['x', 'y']].mean().values
    gt_centroid = matched_gt[['x', 'y']].mean().values
    
    traj_centered = matched_traj[['x', 'y']].values - traj_centroid
    gt_centered = matched_gt[['x', 'y']].values - gt_centroid
    
    H = traj_centered.T @ gt_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    t = gt_centroid - R @ traj_centroid
    
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2] = t
    
    return T, R, t

def create_relative_comparison_matrix(all_results, methods=['avg', 'r', 'rr']):
    """Create matrix comparing methods against each other"""
    comparison_matrix = {}
    
    for group_data in all_results:
        group = group_data['group']
        comparison_matrix[group] = {}
        
        # Compare each method pair
        for i, method1 in enumerate(methods):
            for method2 in methods:
                if method1 != method2:
                    key = f"{method1}_vs_{method2}"
                    
                    # Get RMSE values for comparison
                    rmse1 = group_data.get(f"{method1}_rmse", None)
                    rmse2 = group_data.get(f"{method2}_rmse", None)
                    
                    if rmse1 is not None and rmse2 is not None and rmse2 != 0:
                        improvement_ratio = (rmse2 - rmse1) / rmse2  # Positive means method1 is better
                        comparison_matrix[group][key] = {
                            'rmse_ratio': rmse1 / rmse2,
                            'improvement_percentage': improvement_ratio * 100,
                            'absolute_difference': rmse1 - rmse2
                        }
                    else:
                        comparison_matrix[group][key] = None
    
    return comparison_matrix

def resample(df, length, x_col='x', y_col='y'):
    """Resample dataframe to specified length using interpolation"""
    df = df.dropna(subset=[x_col, y_col])
    if len(df) < 2:
        return df
    
    return pd.DataFrame({
        'x': np.interp(np.linspace(0, len(df)-1, length), np.arange(len(df)), df[x_col]),
        'y': np.interp(np.linspace(0, len(df)-1, length), np.arange(len(df)), df[y_col])
    })

# Set folder path
folder_path = "./output/eval/"
all_csvs = glob.glob(os.path.join(folder_path, "*.csv"))

gt_files = [f for f in all_csvs if f.endswith("gt.csv")]
trajectory_files = [f for f in all_csvs if f not in gt_files]

# Group by prefix
grouped = {}
for gt in gt_files:
    prefix = os.path.basename(gt)[:-6]  # remove 'gt.csv'
    grouped[prefix] = {
        "gt": gt,
        "trajectories": [f for f in trajectory_files if os.path.basename(f).startswith(prefix)]
    }

# Choose matching method: 'closest_distance' or 'same_points'
MATCHING_METHOD = 'closest_distance'

results = []
plot_data = []
transformation_matrices = {}

for prefix, files in grouped.items():
    print(f"Processing group: {prefix}")
    traj_files = files["trajectories"]
    abcde_files = [f for f in traj_files if os.path.basename(f)[-5] in ['a', 'b', 'c', 'd', 'e']]
    r_file = next((f for f in traj_files if f.endswith("r.csv")), None)
    rr_file = next((f for f in traj_files if f.endswith("rr.csv")), None)

    df_gt = pd.read_csv(files["gt"])
    df_gt = df_gt.dropna(subset=['X', 'Y'])
    df_gt = df_gt.rename(columns={'X': 'x', 'Y': 'y'})
    
    if len(df_gt) < 2:
        print(f"Skipping group {prefix}: GT too short.")
        continue

    def prepare_and_eval_comprehensive(traj_file, gt_df, method_suffix):
        if not traj_file:
            return None, None, None, None
            
        df = pd.read_csv(traj_file)
        df = df.dropna(subset=['x', 'y'])
        if len(df) < 2:
            return None, None, None, None
        
        metrics, matched_traj, matched_gt = compute_comprehensive_metrics(df, gt_df, MATCHING_METHOD)
        T, R, t = create_transformation_matrix(matched_traj, matched_gt)
        
        print(f"  {method_suffix} - Points: {metrics['num_points']}, "
              f"RMSE: {metrics['rmse']:.4f}, Hausdorff: {metrics['hausdorff_distance']:.4f}")
        
        return matched_traj, metrics, T, {'rotation': R, 'translation': t}

    # Process average of a-e files
    avg_df, avg_metrics, avg_T, avg_transform = None, None, None, None
    if abcde_files:
        print(f"  Processing {len(abcde_files)} a-e files")
        dfs = []
        for f in abcde_files:
            df = pd.read_csv(f).dropna(subset=['x', 'y'])
            dfs.append(df[['x', 'y']])
        
        if dfs:
            max_len = max(len(df) for df in dfs)
            dfs_resampled = [resample(df, max_len) for df in dfs]
            
            concat_df = pd.concat(dfs_resampled, axis=1)
            avg_df_raw = pd.DataFrame({
                'x': concat_df.filter(like='x').mean(axis=1),
                'y': concat_df.filter(like='y').mean(axis=1)
            }).dropna()
            
            if len(avg_df_raw) >= 2:
                avg_metrics, matched_traj, matched_gt = compute_comprehensive_metrics(avg_df_raw, df_gt, MATCHING_METHOD)
                avg_T, R, t = create_transformation_matrix(matched_traj, matched_gt)
                avg_transform = {'rotation': R, 'translation': t}
                avg_df = matched_traj

    # Process r and rr files
    df_r, r_metrics, r_T, r_transform = prepare_and_eval_comprehensive(r_file, df_gt, "r")
    df_rr, rr_metrics, rr_T, rr_transform = prepare_and_eval_comprehensive(rr_file, df_gt, "rr")

    # Store transformation matrices
    transformation_matrices[prefix] = {
        'avg': {'matrix': avg_T, 'components': avg_transform} if avg_T is not None else None,
        'r': {'matrix': r_T, 'components': r_transform} if r_T is not None else None,
        'rr': {'matrix': rr_T, 'components': rr_transform} if rr_T is not None else None
    }

    # Store plot data
    plot_data.append({
        'prefix': prefix,
        'avg_df': avg_df,
        'df_r': df_r,
        'df_rr': df_rr,
        'df_gt': df_gt
    })

    # Save results with comprehensive metrics
    result_entry = {"group": prefix, "matching_method": MATCHING_METHOD}
    for name, metrics in [("avg", avg_metrics), ("r", r_metrics), ("rr", rr_metrics)]:
        if metrics:
            for k, v in metrics.items():
                result_entry[f"{name}_{k}"] = v
    
    results.append(result_entry)

# Create relative comparison matrix
comparison_matrix = create_relative_comparison_matrix(results)

# --- PLOTTING (updated to force square axes and larger legend/ticks) ---
num_plots = len(plot_data)
if num_plots > 0:
    cols = min(3, num_plots)
    rows = (num_plots + cols - 1) // cols

    # Make each subplot area square-ish by using equal width/height per slot
    # Increase base font size for readability
    plt.rcParams.update({'font.size': 18})

    # Use same size for width/height blocks (6x6 per subplot)
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
    if num_plots == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)

    axes_flat = np.array(axes).flatten()

    for i, data in enumerate(plot_data):
        ax = axes_flat[i]

        if data['avg_df'] is not None:
            ax.plot(data['avg_df']['x'], data['avg_df']['y'], label="Our Method", linewidth=2)
        if data['df_rr'] is not None:
            ax.plot(data['df_rr']['x'], data['df_rr']['y'], label="Small_gicp", linestyle='--')
        if data['df_gt'] is not None:
            ax.plot(data['df_gt']['x'], data['df_gt']['y'], label="Ground truth", color='black', linewidth=2)

        ax.set_title(f"{data['prefix']}", fontsize=26)
        ax.set_xlabel("X", fontsize=22)
        ax.set_ylabel("Y", fontsize=22)

        # Keep equal scaling and force square shape
        ax.set_aspect('equal', adjustable='box')
        # set_box_aspect(1) enforces a square drawing area regardless of the data limits
        ax.set_box_aspect(1)

        ax.grid(True)
        # Bigger legend font size
        ax.legend(fontsize=20)
        # Bigger tick label numbers
        ax.tick_params(axis='both', which='major', labelsize=18)

    # Hide unused subplots
    for i in range(num_plots, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, f"trajectory_plots_{MATCHING_METHOD}.png"), dpi=900, bbox_inches='tight')
    plt.show()

# Save comprehensive results
results_df = pd.DataFrame(results)
output_csv = os.path.join(folder_path, f"comprehensive_metrics_{MATCHING_METHOD}.csv")
results_df.to_csv(output_csv, index=False)
print(f"\nSaved comprehensive metrics to {output_csv}")

# Save transformation matrices
matrices_output = os.path.join(folder_path, f"transformation_matrices_{MATCHING_METHOD}.json")
matrices_for_json = {}
for prefix, data in transformation_matrices.items():
    matrices_for_json[prefix] = {}
    for method, transform_data in data.items():
        if transform_data is not None:
            matrices_for_json[prefix][method] = {
                'matrix': transform_data['matrix'].tolist(),
                'rotation': transform_data['components']['rotation'].tolist(),
                'translation': transform_data['components']['translation'].tolist()
            }
        else:
            matrices_for_json[prefix][method] = None

with open(matrices_output, 'w') as f:
    json.dump(matrices_for_json, f, indent=2)

# Save comparison matrix
comparison_output = os.path.join(folder_path, f"method_comparison_matrix_{MATCHING_METHOD}.json")
with open(comparison_output, 'w') as f:
    json.dump(comparison_matrix, f, indent=2)

print(f"Saved transformation matrices to {matrices_output}")
print(f"Saved method comparison matrix to {comparison_output}")

# Print summary of new metrics
print(f"\nProcessing complete using {MATCHING_METHOD} matching method.")
print("Comprehensive comparison metrics added:")
print("1. Hausdorff Distance - Maximum distance between closest points")
print("2. Fréchet Distance - Considers ordering of points along curves")
print("3. Path Length Analysis - Compares total trajectory lengths")
print("4. Area Between Curves - Measures enclosed area difference")
print("5. Angular Deviation - Compares trajectory directions")
print("6. Correlation Metrics - X/Y coordinate correlations")
print("7. Centroid Distance - Distance between trajectory centers")
print("8. Bounding Box Metrics - Spatial extent comparison")
print("9. Velocity Analysis - Speed profile comparison")
print("10. Method-vs-Method Comparison Matrix - Relative performance")

# Create summary statistics
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)
for i, result in enumerate(results):
    print(f"\nGroup: {result['group']}")
    for method in ['avg', 'rr']:
        if f"{method}_rmse" in result and result[f"{method}_rmse"] is not None:
            print(f"  {method.upper()} Method:")
            print(f"    RMSE: {result[f'{method}_rmse']:.4f}")
            print(f"    Hausdorff: {result[f'{method}_hausdorff_distance']:.4f}")
            print(f"    Fréchet: {result[f'{method}_frechet_distance']:.4f}")
            print(f"    Path Length Ratio: {result[f'{method}_path_length_ratio']:.4f}")
