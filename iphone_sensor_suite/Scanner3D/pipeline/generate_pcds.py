#!/usr/bin/env python3
"""
Generate dense per-capture point clouds by deprojecting depth at higher resolution.

The iOS app saves depth at 256x192 (LiDAR native). This script upsamples depth
to a configurable resolution (default: 960x720, 4x the depth resolution) and
deprojects to get denser point clouds with per-pixel RGB color.

Usage:
    # Generate per-capture point clouds in base frame using LiDAR 3D calibration
    python3 generate_pcds.py --scan-dir ../scan/2026-03-23--19-06-03 \
        --calibration 3d --scale 4

    # Use full RGB resolution (1920x1440) — very dense but slow
    python3 generate_pcds.py --scan-dir ../scan/2026-03-23--19-06-03 \
        --calibration 3d --scale rgb

    # Only world frame (no marker/base transform)
    python3 generate_pcds.py --scan-dir ../scan/2026-03-23--19-06-03 --world-only

Output:
    <scan-dir>/per_capture_dense/capture_NNNNNN_world.ply
    <scan-dir>/per_capture_dense/capture_NNNNNN_marker.ply  (if calibration specified)
    <scan-dir>/per_capture_dense/capture_NNNNNN_base.ply    (if T_base_marker.txt exists)
"""

import argparse
import json
import os
import glob

import cv2
import numpy as np
import open3d as o3d


def get_optical_to_body():
    """Optical frame (X-right, Y-down, Z-forward) -> ARKit body (X-right, Y-up, Z-backward)."""
    return np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1],
    ], dtype=np.float64)


def deproject_capture(capture_dir, target_w, target_h, min_depth=0.1, max_depth=5.0):
    """Deproject depth to 3D points at target resolution with RGB color.

    Args:
        capture_dir: path to capture directory
        target_w, target_h: target resolution for deprojection
        min_depth, max_depth: depth range filter in meters

    Returns:
        points: (N, 3) float64 in optical frame
        colors: (N, 3) float64 in [0, 1]
    """
    # Load depth (256x192, 16-bit mm)
    depth_mm = cv2.imread(os.path.join(capture_dir, 'depth.png'), cv2.IMREAD_UNCHANGED)
    depth_m = depth_mm.astype(np.float32) / 1000.0
    depth_h, depth_w = depth_m.shape

    # Load RGB
    rgb = cv2.cvtColor(cv2.imread(os.path.join(capture_dir, 'rgb.png')), cv2.COLOR_BGR2RGB)
    rgb_h, rgb_w = rgb.shape[:2]

    # Load intrinsics (at RGB resolution)
    with open(os.path.join(capture_dir, 'camera_info.json')) as f:
        cam_info = json.load(f)
    K = np.array(cam_info['k']).reshape(3, 3)

    # Load meta for original resolutions
    with open(os.path.join(capture_dir, 'meta.json')) as f:
        meta = json.load(f)
    orig_rgb_w = meta.get('rgb_width', rgb_w)
    orig_rgb_h = meta.get('rgb_height', rgb_h)

    # Upsample depth to target resolution using linear interpolation for smooth surfaces
    # (INTER_NEAREST produces blocky artifacts; INTER_LINEAR smooths depth transitions)
    depth_upsampled = cv2.resize(depth_m, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Resize RGB to target resolution
    rgb_resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Scale intrinsics from RGB resolution to target resolution
    K_target = K.copy()
    K_target[0, 0] *= target_w / orig_rgb_w
    K_target[0, 2] *= target_w / orig_rgb_w
    K_target[1, 1] *= target_h / orig_rgb_h
    K_target[1, 2] *= target_h / orig_rgb_h

    fx, fy = K_target[0, 0], K_target[1, 1]
    cx, cy = K_target[0, 2], K_target[1, 2]

    # Deproject all pixels
    u, v = np.meshgrid(np.arange(target_w), np.arange(target_h))
    z = depth_upsampled
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = rgb_resized.reshape(-1, 3).astype(np.float64) / 255.0

    # Filter invalid depth
    z_flat = z.reshape(-1)
    valid = (z_flat > min_depth) & (z_flat < max_depth) & np.isfinite(z_flat)
    points = points[valid].astype(np.float64)
    colors = colors[valid]

    return points, colors


def main():
    parser = argparse.ArgumentParser(
        description='Generate dense per-capture point clouds')
    parser.add_argument('--scan-dir', type=str, required=True,
                        help='Scan session directory')
    parser.add_argument('--calibration', type=str, default=None,
                        choices=['solvepnp', '3d', 'triangulate'],
                        help='Calibration method for marker/base frame (default: none)')
    parser.add_argument('--scale', type=str, default='4',
                        help='Depth upsampling: integer scale factor or "rgb" for full RGB resolution (default: 4)')
    parser.add_argument('--min-depth', type=float, default=0.1,
                        help='Min depth in meters (default: 0.1)')
    parser.add_argument('--max-depth', type=float, default=5.0,
                        help='Max depth in meters (default: 5.0)')
    parser.add_argument('--world-only', action='store_true',
                        help='Only generate world frame PLYs')

    args = parser.parse_args()
    scan_dir = os.path.abspath(args.scan_dir)

    captures = sorted([d for d in glob.glob(os.path.join(scan_dir, 'capture_*'))
                      if os.path.isdir(d)])
    if not captures:
        print(f"No captures found in {scan_dir}")
        return

    # Determine target resolution
    # Read first capture's depth to get native resolution
    sample_depth = cv2.imread(os.path.join(captures[0], 'depth.png'), cv2.IMREAD_UNCHANGED)
    depth_h, depth_w = sample_depth.shape
    with open(os.path.join(captures[0], 'meta.json')) as f:
        meta = json.load(f)
    rgb_w = meta.get('rgb_width', 1920)
    rgb_h = meta.get('rgb_height', 1440)

    if args.scale == 'rgb':
        target_w, target_h = rgb_w, rgb_h
    else:
        scale = int(args.scale)
        target_w, target_h = depth_w * scale, depth_h * scale

    print(f"Depth native: {depth_w}x{depth_h}, RGB: {rgb_w}x{rgb_h}")
    print(f"Target resolution: {target_w}x{target_h} ({target_w*target_h} pixels)")
    print(f"Depth range: {args.min_depth}-{args.max_depth}m")
    print(f"Found {len(captures)} captures")

    # Load transforms if needed
    T_optical_to_body = get_optical_to_body()
    T_world_marker = None
    T_base_marker = None

    if not args.world_only and args.calibration:
        twm_path = os.path.join(scan_dir, f'T_world_marker_{args.calibration}.txt')
        if os.path.exists(twm_path):
            T_world_marker = np.loadtxt(twm_path)
            print(f"Using calibration: {args.calibration} ({os.path.basename(twm_path)})")
        else:
            print(f"Warning: {twm_path} not found, skipping marker/base frame")

        tbm_path = os.path.join(scan_dir, 'T_base_marker.txt')
        if os.path.exists(tbm_path):
            T_base_marker = np.loadtxt(tbm_path)

    # Output directory
    out_dir = os.path.join(scan_dir, 'per_capture_dense')
    os.makedirs(out_dir, exist_ok=True)

    for cap_dir in captures:
        name = os.path.basename(cap_dir)
        pose_path = os.path.join(cap_dir, 'T_world_camera.txt')
        if not os.path.exists(pose_path):
            print(f"  {name}: no T_world_camera.txt, skipped")
            continue

        # Deproject at target resolution
        points, colors = deproject_capture(
            cap_dir, target_w, target_h, args.min_depth, args.max_depth)

        T_world_body = np.loadtxt(pose_path)
        T_world_optical = T_world_body @ T_optical_to_body

        # Transform to world frame
        pts_h = np.hstack([points, np.ones((len(points), 1))])
        pts_world = (T_world_optical @ pts_h.T).T[:, :3]

        # Save world PLY
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_world)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        world_path = os.path.join(out_dir, f'{name}_world.ply')
        o3d.io.write_point_cloud(world_path, pcd)

        status = f"  {name}: {len(points)} points -> world"

        # Transform to marker frame
        if T_world_marker is not None:
            T_marker_world = np.linalg.inv(T_world_marker)
            pcd_marker = o3d.geometry.PointCloud(pcd)
            pcd_marker.transform(T_marker_world)
            marker_path = os.path.join(out_dir, f'{name}_marker.ply')
            o3d.io.write_point_cloud(marker_path, pcd_marker)
            status += " + marker"

            # Transform to base frame
            if T_base_marker is not None:
                pcd_base = o3d.geometry.PointCloud(pcd_marker)
                pcd_base.transform(T_base_marker)
                base_path = os.path.join(out_dir, f'{name}_base.ply')
                o3d.io.write_point_cloud(base_path, pcd_base)
                status += " + base"

        print(status)

    print(f"\nSaved to {out_dir}")


if __name__ == '__main__':
    main()
