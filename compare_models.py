r"""
compare_models.py
Automated benchmark comparing all YOLOv11-seg trained runs in D:\New width prj.
Evaluates: Box mAP50-95, Mask mAP50, Mask mAP50-95, Class Recall, and Inference Speed.
"""

from pathlib import Path
import pandas as pd
from ultralytics import YOLO

PROJECT_DIR = Path(r"D:\New width prj")
RUNS_DIR = PROJECT_DIR / "runs" / "segment"
DATA_YAML = PROJECT_DIR / "data.yaml"


def benchmark_models():
    records = []
    # Discover all train directories that have best.pt
    model_dirs = sorted([d for d in RUNS_DIR.glob("train*") if (d / "weights" / "best.pt").exists()])

    if not model_dirs:
        print("No trained models found in runs/segment/")
        return

    print(f"Discovered {len(model_dirs)} trained models. Commencing evaluation on {DATA_YAML}...\n")

    for m_dir in model_dirs:
        model_name = m_dir.name
        weights_path = m_dir / "weights" / "best.pt"
        print(f"--- Evaluating {model_name} ({weights_path.name}) ---")

        try:
            model = YOLO(str(weights_path))
            metrics = model.val(
                data=str(DATA_YAML),
                imgsz=640,
                batch=16,
                conf=0.22,
                iou=0.45,
                device=0,
                verbose=False,
            )

            # Global Metrics
            box_map50 = metrics.box.map50
            box_map = metrics.box.map
            mask_map50 = metrics.seg.map50
            mask_map = metrics.seg.map
            speed_ms = metrics.speed.get("inference", 0.0)

            records.append({
                "Run": model_name,
                "Mask mAP50": round(float(mask_map50), 3),
                "Mask mAP50-95": round(float(mask_map), 3),
                "Box mAP50": round(float(box_map50), 3),
                "Box mAP50-95": round(float(box_map), 3),
                "Inference (ms)": round(float(speed_ms), 2),
                "FPS": round(1000.0 / max(float(speed_ms), 1e-4), 1),
            })

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")

    df = pd.DataFrame(records)
    print("\n========================= BENCHMARK COMPARISON TABLE =========================")
    print(df.to_string(index=False))
    print("==============================================================================")
    df.to_csv(PROJECT_DIR / "models_benchmark_summary.csv", index=False)
    print(f"Summary saved to: {PROJECT_DIR / 'models_benchmark_summary.csv'}")


if __name__ == "__main__":
    benchmark_models()