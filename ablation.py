#!/usr/bin/env python3
# ablation_runner.py
#
# Run fixed-weight ablations over subfolders of PCD datasets.
# Requires your previously provided code to be importable:
# - process_non_repetitive_lidar_scans (adapted with fixed_weights & freeze_adaptation)
# - create_combined_cloud
#
# If that code is in a separate file/module, change the import below:
# from your_module import process_non_repetitive_lidar_scans, create_combined_cloud
# If it's in the *same* file, you can comment the import and run after the definitions.

import os
import time
import json
import numpy as np
import pandas as pd
import open3d as o3d
from typing import Dict, List, Optional, Tuple

# ---- IMPORTS: change this line if your functions are in another module ----
#from __main__ import process_non_repetitive_lidar_scans, create_combined_cloud  # noqa

# If you're running this as a standalone file separate from your main code,
# replace the above import with:
from nonrep import process_non_repetitive_lidar_scans, create_combined_cloud


# -----------------------
# Dataset discovery
# -----------------------
def find_pcd_subfolders(root_dir: str, min_files: int = 1) -> List[str]:
    """
    Return subfolders (one level deep) that contain at least `min_files` .pcd files.
    """
    subfolders = []
    for entry in os.scandir(root_dir):
        if entry.is_dir():
            pcd_count = sum(
                1
                for f in os.scandir(entry.path)
                if f.is_file() and f.name.lower().endswith(".pcd")
            )
            if pcd_count >= min_files:
                subfolders.append(entry.path)
    return sorted(subfolders)


# -----------------------
# KPI summarizer
# -----------------------
def summarize_run(
    results: List[Dict], final_poses: List[Optional[np.ndarray]], use_xy: bool = True
) -> Dict:
    """
    Compute quick KPIs for a run.
    """
    ok = [r for r in results if "error" not in r]
    n_ok = len(ok)
    n_total = len(results)
    confs = [r.get("prediction_confidence", 0.0) for r in ok]
    avg_conf = float(np.mean(confs)) if confs else 0.0

    traj = [p for p in final_poses if p is not None]
    total_dist = 0.0
    if len(traj) > 1:
        for i in range(1, len(traj)):
            if use_xy:
                dp = np.linalg.norm(traj[i][:2] - traj[i - 1][:2])
            else:
                dp = np.linalg.norm(traj[i][:3] - traj[i - 1][:3])
            total_dist += float(dp)

    return {
        "success_scans": n_ok,
        "total_scans": n_total,
        "avg_prediction_conf": avg_conf,
        "total_distance_m": total_dist,
    }


# -----------------------
# Save artifacts for each run
# -----------------------
def save_run_artifacts(
    out_root: str,
    dataset_name: str,
    ablation_name: str,
    results: List[Dict],
    final_poses: List[Optional[np.ndarray]],
    combined_cloud: Optional[o3d.geometry.PointCloud],
):
    """
    Save per-run CSV (poses) and JSON (results), plus combined map (if provided).
    """
    run_dir = os.path.join(out_root, dataset_name, ablation_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save poses CSV
    rows = []
    for i, (r, p) in enumerate(zip(results, final_poses)):
        if p is None:
            continue
        rows.append(
            {
                "scan_index": i,
                "scan_file": r.get("scan_file", f"scan_{i}"),
                "x": float(p[0]),
                "y": float(p[1]),
                "z": float(p[2]),
                "yaw_rad": float(p[3]),
                "yaw_deg": float(np.degrees(p[3])),
                "prediction_confidence": float(r.get("prediction_confidence", 0.0)),
                "z_redistributed": bool(r.get("z_redistributed", False)),
                "feature_w": float(r.get("weights", {}).get("feature", np.nan)),
                "geometric_w": float(r.get("weights", {}).get("geometric", np.nan)),
                "temporal_w": float(r.get("weights", {}).get("temporal", np.nan)),
                "adaptation_frozen": bool(r.get("adaptation_frozen", False)),
            }
        )
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(run_dir, "final_poses.csv"), index=False)

    # Save combined cloud (optional)
    if combined_cloud is not None:
        o3d.io.write_point_cloud(os.path.join(run_dir, "combined_map.pcd"), combined_cloud)

    # Save lightweight results JSON (numpy to lists)
    json_results = []
    for r in results:
        jr = {}
        for k, v in r.items():
            if isinstance(v, np.ndarray):
                jr[k] = v.tolist()
            elif k == "features":
                jr[k] = "extracted" if v else "failed"
            else:
                jr[k] = v
        json_results.append(jr)
    with open(os.path.join(run_dir, "processing_results.json"), "w") as f:
        json.dump(json_results, f, indent=2)


# -----------------------
# Single ablation execution
# -----------------------
def run_single_ablation(
    dataset_path: str,
    ablation_name: str,
    weights: Tuple[float, float, float],
    step: int,
    max_clouds: Optional[int],
    force_z_zero: bool,
    z_method: str,
    visualize: bool,
):
    """
    Runs one ablation on one dataset by calling your process_non_repetitive_lidar_scans
    with fixed weights and frozen adaptation.
    """
    print(f"\n>>>> Ablation: {ablation_name}  weights={weights}")
    t0 = time.time()

    results, pred_poses, obs_poses, final_poses = process_non_repetitive_lidar_scans(
        observation_folder=dataset_path,
        visualize=visualize,
        observation_step_size=step,
        observation_start_index=0,
        max_observation_clouds=max_clouds,
        force_z_zero=force_z_zero,
        z_redistribution_method=z_method,
        # --- ablation controls (must exist in your adapted function) ---
        fixed_weights=weights,
        freeze_adaptation=True,
    )

    runtime = round(time.time() - t0, 3)
    print(f"[{ablation_name}] Runtime: {runtime}s")

    return results, final_poses, runtime


# -----------------------
# Main ablation suite
# -----------------------
def run_ablation_suite(
    root_dir: str,
    output_root: str = "./output/ablation",
    # Default ablations:
    ablations: Dict[str, Tuple[float, float, float]] = None,
    # Other params:
    step: int = 1,
    max_clouds: Optional[int] = None,
    force_z_zero: bool = True,
    z_method: str = "prediction",
    visualize_each_run: bool = False,
):
    """
    Runs the ablation set across every subfolder (dataset) under `root_dir`.

    Outputs:
      - {output_root}/{dataset}/{ablation}/final_poses.csv + processing_results.json (+ combined_map.pcd)
      - {output_root}/summary.csv (one row per dataset × ablation)
    """
    if ablations is None:
        ablations = {
            "feature_only": (1.0, 0.0, 0.0),
            "geometric_only": (0.0, 1.0, 0.0),
            "temporal_only": (0.0, 0.0, 1.0),
        }

    datasets = find_pcd_subfolders(root_dir, min_files=1)
    if not datasets:
        print(f"[WARN] No subfolders with .pcd under: {root_dir}")
        return

    os.makedirs(output_root, exist_ok=True)
    summary_rows = []

    # We'll try to construct a combined map using your provided helper after each run.
    # It relies on Pctools.load_pcd_files which must be available in your env.
    try:
        from Pctools import load_pcd_files
    except Exception:
        load_pcd_files = None
        print("[WARN] Pctools.load_pcd_files not importable; combined maps will be skipped.")

    for ds_path in datasets:
        dataset_name = os.path.basename(ds_path.rstrip(os.sep))
        print(f"\n================ DATASET: {dataset_name} ================")

        for abl_name, w in ablations.items():
            # Run one ablation
            results, final_poses, runtime = run_single_ablation(
                dataset_path=ds_path,
                ablation_name=abl_name,
                weights=w,
                step=step,
                max_clouds=max_clouds,
                force_z_zero=force_z_zero,
                z_method=z_method,
                visualize=visualize_each_run,
            )

            # Build a combined cloud (optional)
            combined_cloud = None
            if load_pcd_files is not None and final_poses and any(p is not None for p in final_poses):
                try:
                    pcds = load_pcd_files(ds_path, step, 0, max_clouds)
                    combined_cloud = create_combined_cloud(pcds, final_poses)
                except Exception as e:
                    print(f"[WARN] Combined map creation failed: {e}")

            # Save artifacts per run
            save_run_artifacts(
                output_root,
                dataset_name,
                abl_name,
                results,
                final_poses,
                combined_cloud,
            )

            # KPIs row
            # Determine if XY-only distance is appropriate (use force_z_zero hint)
            kpis = summarize_run(results, final_poses, use_xy=force_z_zero)
            kpis.update(
                dict(
                    dataset=dataset_name,
                    ablation=abl_name,
                    feature_weight=w[0],
                    geometric_weight=w[1],
                    temporal_weight=w[2],
                    force_z_zero=force_z_zero,
                    z_method=z_method,
                    step_size=step,
                    max_clouds=-1 if (max_clouds is None or max_clouds == 0) else max_clouds,
                    runtime_sec=runtime,
                )
            )
            summary_rows.append(kpis)

    # Save summary CSV
    if summary_rows:
        summary_path = os.path.join(output_root, "summary.csv")
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"\n=== Ablation summary saved to: {summary_path} ===")
    else:
        print("\n[WARN] No runs were completed; summary not saved.")


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run fixed-weight ablations for non-repetitive LiDAR scans.")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Folder that contains subfolders, each with PCD files.")
    parser.add_argument("--output_root", type=str, default="./output/ablation",
                        help="Where to store per-run artifacts and the summary.")
    parser.add_argument("--step", type=int, default=1,
                        help="Observation step size (load every Nth PCD).")
    parser.add_argument("--max_clouds", type=int, default=0,
                        help="Limit number of clouds per dataset (0 = no limit).")
    parser.add_argument("--force_z_zero", action="store_true",
                        help="Enable Z=0 mode with redistribution.")
    parser.add_argument("--z_method", type=str, default="prediction",
                        choices=["prediction", "dominant_axis", "equal"],
                        help="Method for z redistribution when Z=0 is enabled.")
    parser.add_argument("--visualize", action="store_true",
                        help="Show movement plot per run (slower).")

    args = parser.parse_args()

    run_ablation_suite(
        root_dir=args.root_dir,
        output_root=args.output_root,
        step=args.step,
        max_clouds=(None if args.max_clouds == 0 else args.max_clouds),
        force_z_zero=args.force_z_zero,
        z_method=args.z_method,
        visualize_each_run=args.visualize,
    )
