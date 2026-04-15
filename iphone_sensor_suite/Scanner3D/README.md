# Scanner3D

Offline 3D scanning app for iPhone. Capture depth, RGB, camera pose, and confidence maps using the LiDAR sensor, then export scans for post-processing with Python tools.

## iOS App

Download the free iOS scanning app:

*Coming soon on the App Store*

## Features

- Real-time depth heatmap preview at 60fps
- Capture depth (256x192), RGB (1920x1440), intrinsics, camera pose, and confidence maps
- Save and manage multiple scan sessions in the Library
- Export scans as zip files via the iOS Files app
- ArUco marker generation for camera-to-robot calibration
- Mesh scanning mode with ARKit scene reconstruction

## Requirements

- iPhone 12 Pro or newer (with LiDAR sensor)
- iOS 17.5+

## How It Works

1. **Scan** -- Point your iPhone at an object, tap "Start Scan", then tap "Capture" for each frame
2. **Save** -- Save the scan session to the Library
3. **Export** -- Export as a zip file and transfer to your computer via Files app
4. **Process** -- Use the Python tools below to fuse captures into a 3D point cloud or mesh

## Exported Data Format

Each scan session is exported as a zip containing `capture_NNNNNN/` folders. Each capture folder contains:

| File | Description |
|------|-------------|
| `pointcloud.ply` | Point cloud in camera optical frame (confidence-filtered) |
| `depth.png` | 16-bit grayscale depth in millimeters |
| `rgb.png` | Color image |
| `camera_info.json` | Camera intrinsics (ROS CameraInfo format) |
| `T_world_camera.txt` | 4x4 ARKit camera-to-world transform |
| `meta.json` | Capture metadata (timestamps, resolutions) |
| `depth_vis.png` | Depth visualization (red=close, blue=far) |

## Python Tools

Offline fusion tools for combining multiple captures into a single 3D point cloud or mesh.

### Setup

Requires Python 3.11 (Open3D does not support 3.13 yet).

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Usage

```bash
source venv/bin/activate

# Point cloud fusion (from PLY files)
python3 fuse_pointclouds.py --data-dir <scan_dir>

# Point cloud fusion (from depth images)
python3 fuse_pointclouds.py --data-dir <scan_dir> --mode depth

# TSDF mesh fusion (best quality)
python3 fuse_pointclouds.py --data-dir <scan_dir> --mode tsdf --voxel-size 0.002

# Point cloud with ICP refinement
python3 fuse_pointclouds.py --data-dir <scan_dir> --mode ply --icp
```

### Fusion Modes

| Mode | Output | Best For |
|------|--------|----------|
| `--mode ply` (default) | Point cloud (stacked from PLY files) | Quick preview |
| `--mode depth` | Point cloud (from depth images) | Custom depth filtering |
| `--mode ply --icp` | Point cloud (ICP-aligned) | When ARKit pose has drift |
| `--mode tsdf` | Triangle mesh (clean, denoised) | Final quality output |

### Calibration

ArUco-based camera-to-robot calibration tools:

```bash
# Single-frame calibration
python3 calibrate.py --data-dir <scan_dir>

# Multi-frame 3D calibration
python3 calibrate_3d.py --data-dir <scan_dir>

# Triangulation-based calibration
python3 calibrate_triangulate.py --data-dir <scan_dir>
```

### Key Options

| Option | Description |
|--------|-------------|
| `--data-dir` | Directory containing `capture_*` folders (required) |
| `--mode` | `ply`, `depth`, or `tsdf` |
| `--voxel-size` | Voxel size in meters (default: 0.005) |
| `--icp` | Enable ICP alignment refinement (ply/depth modes only) |
| `--bounds` | Workspace bounds filter: `x_min,x_max,y_min,y_max,z_min,z_max` |
| `--output` | Output PLY path |
