# 🧱 Real-Time Computer Vision Bead Metrology for 3D Concrete Printing (3DCP)

[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20ROS2-blue.svg)](https://github.com/Hamdan070905/3D-Printing-Robot-width-measurement-project)
[![Python](https://img.shields.io/badge/Python-3.10-green.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/YOLO-v11--seg%20(train--6)-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Metrology](https://img.shields.io/badge/Metrology-Sub--Pixel%20Parabolic%20%2B%20Schmitt%20Trigger-brightgreen.svg)]()
[![CUDA](https://img.shields.io/badge/Acceleration-PyTorch%20CUDA%20(RTX%204060)-76B900.svg)](https://developer.nvidia.com/cuda-zone)

An industrial-grade, real-time edge computer vision metrology pipeline designed for automated bead width extraction, topological centerline tracing, and closed-loop quality assurance in 3D Concrete Construction Printing (3DCP). 

Built to eliminate geometric deadlocks, toolpath curvature distortions, and high-intensity surface glare on wet cement slurry under high-speed gantry deposition ($2400\text{ mm/min}$).

---

## 📌 Key Architectural Highlights

* **YOLOv11-seg Deep Segmentation (`train-6`):**
  Fine-tuned on production 3DCP bead datasets. Delivers **0.738 Mask mAP50**, **0.648 Mask mAP50-95**, and **0.732 Defect Recall**, eliminating false negative drops on underwidth necking or slurry smudges.
* **Topological Medial Skeletonization & Local Tangents:**
  Replaces fragile global linear PCA projections with morphological skeleton graph tracing. Computes local tangent frames $\mathbf{T}$ and true orthogonal normals $\mathbf{N} \perp \mathbf{T}$ across arbitrary geometries (straight toolpaths, tight $90^\circ$ turns, circular pots, and U-turns).
* **Sub-Pixel Parabolic Ridge Refinement:**
  Applies Euclidean distance transform ($L_2$) paired with 3-point sub-pixel parabolic peak interpolation to prevent discrete pixel-grid snapping:
  $$\delta = \frac{d_{-1} - d_{+1}}{2(d_{-1} - 2d_0 + d_{+1})}$$
* **Worst-Case Multi-Node Defect Voting:**
  Eliminates the industrial vulnerability where arithmetic mean width masks localized bulging or necking. If $\ge 2$ nodes breach tolerance thresholds, the system flags `OVER_DEFECT` or `UNDER_DEFECT`.
* **Schmitt Trigger Limit-Cycle Hysteresis:**
  A dual-threshold $\pm 0.8\text{ mm}$ deadband prevents boundary chatter on PLC telemetry channels when bead dimensions fluctuate near specification limits.
* **Non-Blocking Threaded Capture:**
  Dedicated background thread ingestion for USB action cameras (e.g., SJCAM) that flushes stale OS buffers and ensures inference runs on the freshest physical frame with zero latency drift.
* **Dynamic Optical Auto-Scaling (ArUco Fiducial):**
  Auto-calibrates scale ($\text{mm/px}$) via head-mounted ArUco markers with a Kalman filter tracker that coasts smoothly through nozzle shadows.

---

## 📊 Metrology vs. Traditional Approaches

| Feature | Baseline Vision / Naive PCA | Kelvin 3D Production Pipeline |
| :--- | :--- | :--- |
| **Curved & U-Turn Beads** | Fails (linear axis cuts across void) | **Morphological Medial Skeleton + Tangent Spline** |
| **Grid Quantization** | Coarse ($\pm 1\text{ px}$ step jitter) | **Sub-Pixel Parabolic Peak Refinement** |
| **Edge Anchor Integrity** | Cartesian median floats outside moving bead | **Decoupled: Instantaneous Perimeter Intersections** |
| **Defect Detection** | Arithmetic Mean (Masks local bulges) | **Worst-Case Multi-Node Voting ($\ge 2$ nodes)** |
| **Boundary Stability** | Hard thresholding (Status chatters) | **Schmitt Trigger Hysteresis ($\pm 0.8\text{ mm}$ deadband)** |
| **Camera Ingestion** | Synchronous blocking read (USB ghost lag) | **Threaded Non-Blocking Buffer Flush** |

---

## 🛠️ System Architecture

```text
       [ USB Action Camera / SJCAM ]
                     │
         (Non-Blocking Thread Ingestion)
                     ▼
          [ Preprocessing: CLAHE ]
                     │
          [ YOLOv11-seg Inference ]  ──► (Retina Mask Extraction)
                     │
        [ Temporal EMA + Coasting ]
                     │
      [ Morphological Skeleton Tracing ]
                     │
     [ Local Tangents & 90° Normal Cuts ]
                     │
     [ Sub-Pixel Parabolic Refinement ]
                     │
   ┌─────────────────┴─────────────────┐
   ▼                                   ▼
[ Visual Badged HUD ]      [ Quality Telemetry Engine ]
• W1-W10 Staggered Badges  • Schmitt Trigger Hysteresis
• Real-Time Spine Display  • Multi-Node Defect Voting
• Perimeter Snap Anchors   • CSV Audit Stream (Time, FPS, mm)


├── bead_stabilizer.py            # Core metrology: skeleton extraction, tangents, parabolic refinement
├── live_bead_analyzer.py         # Production launcher: threaded video, UI badges, defect voting, telemetry
├── compare_models.py             # Validation benchmark suite across training iterations
├── models_benchmark_summary.csv  # Precision, Recall, and mAP metrics
├── run_live.bat                  # One-click factory deployment script
├── concrete_metrology_ws/        # ROS2 workspace package (bead_metrology_node)
│   └── src/bead_metrology/
│       ├── bead_metrology/
│       ├── config/params.yaml
│       └── launch/bead_metrology.launch.py
├── runs/segment/train-6/weights/ # Production weights
│   └── best.pt                   # Tuned YOLOv11-seg production model (~5.2 MB)
├── data.yaml                     # Dataset taxonomy and class definitions
└── .gitignore                    # Excludes heavy video datasets and frame caches
