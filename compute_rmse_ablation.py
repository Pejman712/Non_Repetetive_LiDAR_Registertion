#!/usr/bin/env python3
"""
Compute closest-point RMSEs and produce trajectory plots for a directory layout:

root/
├── <AREA>/
│   ├── feature_only/final_poses.csv
│   ├── geometric_only/final_poses.csv
│   └── temporal_only/final_poses.csv
└── gt/
    └── <AREA>gt.csv

Usage:
    python compute_rmse_ablation_plot.py --root <path> --output summary.csv
Options:
    --plots-dir <dir>   Directory for plots (default: <root>/plots)
    --no-plot           Skip plot generation
    --verbose           Print detailed progress and saved file paths
"""

import argparse
import os
import sys
import math
import numpy as np
import pandas as pd

# Force headless plotting backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------- Data Reading with Cleaning ---------------------- #
def _read_points_csv(path: str):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if not (('x' in cols and 'y' in cols) or ('X' in df.columns and 'Y' in df.columns)):
        raise ValueError(f"{path}: expected x/y columns but found {list(df.columns)}")

    cx = cols.get('x', 'X')
    cy = cols.get('y', 'Y')
    cz = cols.get('z') or ('Z' if 'Z' in df.columns else None)

    if cz is None:
        df["_Z_TMP_"] = 0.0
        cz = "_Z_TMP_"

    used = df[[cx, cy, cz]].apply(pd.to_numeric, errors="coerce")
    before = len(used)

    # Drop NaN and Inf
    used = used.replace([np.inf, -np.inf], np.nan).dropna()
    dropped = before - len(used)
    if dropped > 0:
        print(f"[WARN] {path}: dropped {dropped} non-finite rows out of {before}", file=sys.stderr)

    xy = used[[cx, cy]].to_numpy(dtype=float, copy=False)
    xyz = used[[cx, cy, cz]].to_numpy(dtype=float, copy=False)
    return {"xy": xy, "xyz": xyz, "df": used}

# ---------------------- RMSE Computation ---------------------- #
def _closest_rmse(A: np.ndarray, B: np.ndarray) -> float:
    if A.size == 0 or B.size == 0:
        return math.nan
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(B)
        dists, _ = tree.query(A, k=1, workers=-1)
    except Exception:
        diffs = A[:, None, :] - B[None, :, :]
        dists = np.linalg.norm(diffs, axis=2).min(axis=1)
    return float(np.sqrt(np.mean(dists ** 2)))

def compute_pairwise_rmses(gt_pts, est_pts):
    res = {}
    for key in ("xy", "xyz"):
        gt = gt_pts[key]
        es = est_pts[key]
        rmse_gt2est = _closest_rmse(gt, es)
        rmse_est2gt = _closest_rmse(es, gt)
        rmse_sym = np.nanmean([rmse_gt2est, rmse_est2gt])
        res[f"rmse_{key}_gt_to_est"] = rmse_gt2est
        res[f"rmse_{key}_est_to_gt"] = rmse_est2gt
        res[f"rmse_{key}_chamfer"] = rmse_sym
        res[f"n_{key}_gt"] = gt.shape[0]
        res[f"n_{key}_est"] = es.shape[0]
    return res

# ---------------------- Plotting Helpers ---------------------- #
def _extract_xy_ordered(df: pd.DataFrame) -> np.ndarray:
    cols = {c.lower(): c for c in df.columns}
    cx = cols.get('x', 'X' if 'X' in df.columns else None)
    cy = cols.get('y', 'Y' if 'Y' in df.columns else None)
    if cx is None or cy is None:
        raise ValueError("CSV missing x/y columns")
    order_col = None
    for k in ['scan_index', 'time', 'timestamp', 'frame', 'index']:
        if k in cols:
            order_col = cols[k]
            break
    if order_col is not None:
        df = df.sort_values(order_col)
    return df[[cx, cy]].to_numpy(dtype=float, copy=False)

def _extract_z_ordered(df: pd.DataFrame) -> np.ndarray:
    cols = {c.lower(): c for c in df.columns}
    cz = cols.get('z', 'Z' if 'Z' in df.columns else None)
    if cz is None:
        return np.zeros((len(df),), dtype=float)
    order_col = None
    for k in ['scan_index', 'time', 'timestamp', 'frame', 'index']:
        if k in cols:
            order_col = cols[k]
            break
    if order_col is not None:
        df = df.sort_values(order_col)
    return df[cz].to_numpy(dtype=float, copy=False)

def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)

def plot_trajectories(gt_csv: str, est_csv: str, out_dir: str, title_prefix: str, verbose: bool=False):
    os.makedirs(out_dir, exist_ok=True)
    gt_df = pd.read_csv(gt_csv)
    est_df = pd.read_csv(est_csv)

    gt_xy = _extract_xy_ordered(gt_df)
    est_xy = _extract_xy_ordered(est_df)

    base = _safe_name(title_prefix)

    # XY plot
    fig = plt.figure()
    if gt_xy.size:
        plt.plot(gt_xy[:,0], gt_xy[:,1], label="GT")
    if est_xy.size:
        plt.plot(est_xy[:,0], est_xy[:,1], label="Estimate")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(f"{title_prefix} — XY Trajectory")
    plt.legend()
    plt.grid(True)
    xy_path = os.path.join(out_dir, f"{base}_xy.png")
    fig.savefig(xy_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    if verbose:
        print(f"[PLOT] Saved: {xy_path}")

    # Z vs index
    gt_z = _extract_z_ordered(gt_df)
    est_z = _extract_z_ordered(est_df)
    fig2 = plt.figure()
    if gt_z.size:
        plt.plot(np.arange(len(gt_z)), gt_z, label="GT")
    if est_z.size:
        plt.plot(np.arange(len(est_z)), est_z, label="Estimate")
    plt.xlabel("Index")
    plt.ylabel("Z (m)")
    plt.title(f"{title_prefix} — Z vs Index")
    plt.legend()
    plt.grid(True)
    z_path = os.path.join(out_dir, f"{base}_z.png")
    fig2.savefig(z_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    if verbose:
        print(f"[PLOT] Saved: {z_path}")

    return {"xy": xy_path, "z": z_path}

# ---------------------- Processing ---------------------- #
def find_areas(root: str):
    return [n for n in sorted(os.listdir(root)) if n != "gt" and os.path.isdir(os.path.join(root, n))]

def area_to_gt_filename(area: str) -> str:
    return f"{area}gt.csv"

def process_root(root: str, out_csv: str, plots_dir: str, make_plots: bool, verbose: bool=False) -> pd.DataFrame:
    gt_dir = os.path.join(root, "gt")
    if not os.path.isdir(gt_dir):
        raise SystemExit(f"Missing gt directory at: {gt_dir}")

    variants = ("feature_only", "geometric_only", "temporal_only")
    rows = []
    os.makedirs(plots_dir, exist_ok=True)

    for area in find_areas(root):
        area_path = os.path.join(root, area)
        gt_file = os.path.join(gt_dir, area_to_gt_filename(area))
        if not os.path.isfile(gt_file):
            print(f"[WARN] GT file not found for area '{area}': {gt_file}", file=sys.stderr)
            continue

        try:
            gt_pts = _read_points_csv(gt_file)
        except Exception as e:
            print(f"[ERROR] reading GT for {area}: {e}", file=sys.stderr)
            continue

        for variant in variants:
            fp = os.path.join(area_path, variant, "final_poses.csv")
            if not os.path.isfile(fp):
                print(f"[WARN] Missing {fp}", file=sys.stderr)
                continue
            try:
                est_pts = _read_points_csv(fp)
                metrics = compute_pairwise_rmses(gt_pts, est_pts)
            except Exception as e:
                print(f"[ERROR] processing {fp}: {e}", file=sys.stderr)
                continue

            if make_plots:
                try:
                    subdir = os.path.join(plots_dir, area)
                    os.makedirs(subdir, exist_ok=True)
                    title = f"{area} {variant}"
                    plot_trajectories(gt_file, fp, subdir, title, verbose=verbose)
                except Exception as e:
                    print(f"[WARN] Plotting failed for {area}/{variant}: {e}", file=sys.stderr)

            row = {"area": area, "variant": variant}
            row.update(metrics)
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("[WARN] No results found. Did you point --root to the ablation folder?", file=sys.stderr)
    else:
        preferred = [
            "area","variant",
            "rmse_xy_gt_to_est","rmse_xy_est_to_gt","rmse_xy_chamfer",
            "rmse_xyz_gt_to_est","rmse_xyz_est_to_gt","rmse_xyz_chamfer",
            "n_xy_gt","n_xy_est","n_xyz_gt","n_xyz_est"
        ]
        cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
        df = df[cols].sort_values(["area","variant"]).reset_index(drop=True)
        out_path = os.path.join(root, out_csv)
        df.to_csv(out_path, index=False)
        if verbose:
            print(f"[CSV] Wrote summary to: {out_path}")
        else:
            print(f"Wrote summary to: {out_path}")
    return df

# ---------------------- Main ---------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".", help="Root directory containing areas and a 'gt' folder")
    parser.add_argument("--output", type=str, default="summary.csv", help="Output CSV filename")
    parser.add_argument("--plots-dir", type=str, default=None, help="Directory for plots (default: <root>/plots)")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")
    args = parser.parse_args()

    plots_dir = args.plots_dir or os.path.join(os.path.abspath(args.root), "plots")
    make_plots = not args.no_plot
    process_root(os.path.abspath(args.root), args.output, plots_dir=plots_dir, make_plots=make_plots, verbose=args.verbose)

if __name__ == "__main__":
    main()
