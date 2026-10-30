# 3DCP Bead Metrology ROS 2 Package

A modular, production-ready ROS 2 package for real-time 3D Concrete Printing (3DCP) bead width metrology, monotonic horizontal scanline slicing ($W_1$ to $W_{10}$), and multi-class defect classification using YOLOv11n-seg (`train-4` weights).

Tested across Ubuntu 22.04 / 24.04 (ROS 2 Humble, Iron, and Jazzy).

---

## 1. Workspace Directory Layout

```
concrete_metrology_ws/
└── src/
    └── bead_metrology/
        ├── CMakeLists.txt / setup.py
        ├── package.xml
        ├── setup.cfg
        ├── config/
        │   └── params.yaml
        ├── launch/
        │   └── bead_metrology.launch.py
        ├── resource/
        │   └── bead_metrology
        └── bead_metrology/
            ├── __init__.py
            └── bead_metrology_node.py
```

---

## 2. Installation & Prerequisites

On your Ubuntu Linux robot host or container:

```bash
# 1. Source ROS 2 (e.g., Humble or Jazzy)
source /opt/ros/${ROS_DISTRO}/setup.bash

# 2. Install ROS 2 dependencies
sudo apt-get update && sudo apt-get install -y \
  ros-${ROS_DISTRO}-cv-bridge \
  ros-${ROS_DISTRO}-sensor-msgs \
  ros-${ROS_DISTRO}-std-msgs \
  ros-${ROS_DISTRO}-geometry-msgs \
  python3-colcon-common-extensions

# 3. Install Python CV & ML dependencies
pip3 install ultralytics opencv-python numpy
```

---

## 3. Build & Execution

```bash
# Navigate to the workspace root
cd ~/concrete_metrology_ws

# Copy or link your trained train-4 weights (e.g., best.pt)
mkdir -p weights
cp /path/to/train-4/weights/best.pt weights/best.pt

# Build the package
colcon build --symlink-install --packages-select bead_metrology

# Source the overlay
source install/setup.bash

# Run using the ROS 2 Launch system
ros2 launch bead_metrology bead_metrology.launch.py
```

---

## 4. Topics & Telemetry Interface

| Topic Name | Message Type | Purpose |
| :--- | :--- | :--- |
| `/gantry_cam/image_raw` | `sensor_msgs/msg/Image` | Camera input stream (RGB/BGR) |
| `/3dcp/metrology_telemetry` | `std_msgs/msg/String` (JSON) | Metrology payload with slice widths ($W_1 \dots W_{10}$), defect alerts, confidence, and latency |
| `/3dcp/annotated_image` | `sensor_msgs/msg/Image` | Processed visualization frame with mask overlay, medial axis, and scanlines |

### Sample JSON Telemetry Payload (`/3dcp/metrology_telemetry`):

```json
{
  "header": {
    "frame_id": "gantry_cam_optical_frame",
    "stamp_sec": 1774160000,
    "stamp_nanosec": 123456789,
    "frame_index": 420
  },
  "classification": {
    "class_id": 0,
    "class_name": "Normal",
    "confidence": 0.892,
    "track_id": 1
  },
  "metrology": {
    "mean_width_mm": 47.85,
    "std_width_mm": 2.14,
    "valid_slices": 10,
    "total_slices": 10,
    "defect_alert": "NONE",
    "slices": [
      {"node": 1, "width_mm": 48.2, "status": "NORMAL"},
      {"node": 2, "width_mm": 49.1, "status": "NORMAL"},
      {"node": 3, "width_mm": 47.9, "status": "NORMAL"},
      {"node": 4, "width_mm": 46.8, "status": "NORMAL"},
      {"node": 5, "width_mm": 47.4, "status": "NORMAL"},
      {"node": 6, "width_mm": 48.0, "status": "NORMAL"},
      {"node": 7, "width_mm": 47.6, "status": "NORMAL"},
      {"node": 8, "width_mm": 48.5, "status": "NORMAL"},
      {"node": 9, "width_mm": 47.1, "status": "NORMAL"},
      {"node": 10, "width_mm": 47.9, "status": "NORMAL"}
    ]
  },
  "performance": {
    "inference_fps": 28.4,
    "latency_ms": 31.25
  }
}
```
