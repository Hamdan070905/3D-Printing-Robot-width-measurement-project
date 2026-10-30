from pathlib import Path
from ultralytics import YOLO

if __name__ == "__main__":
    # All model checkpoints to test (update paths if needed)
    model_runs = {
        "train-1 (Baseline)": r"D:\New width prj\runs\segment\train\weights\best.pt",
        "train-3": r"D:\New width prj\runs\segment\train-3\weights\best.pt",
        "train-4": r"D:\New width prj\runs\segment\train-4\weights\best.pt",
        "train-5 (Current)": r"D:\New width prj\runs\segment\train-5\weights\best.pt",
    }

    data_yaml = "data.yaml"

    print("=" * 65)
    print("3DCP MULTI-MODEL METROLOGY & ACCURACY BENCHMARK")
    print("=" * 65)

    results_summary = {}

    for name, path in model_runs.items():
        model_path = Path(path)
        if not model_path.exists():
            print(f"\n[SKIPPED] {name}: Weights not found at {path}")
            continue
            
        print(f"\nEvaluating Model: {name}...")
        model = YOLO(str(model_path))
        
        # Run validation on the test split (workers=0 prevents Windows multiprocessing errors)
        metrics = model.val(data=data_yaml, split='test', imgsz=640, workers=0, verbose=False)
        
        box_map50 = metrics.box.map50
        seg_map50 = metrics.seg.map50  # Mask mAP@0.5 is critical for bead width accuracy
        
        results_summary[name] = {
            "box_map50": box_map50,
            "seg_map50": seg_map50
        }
        print(f"-> {name} | Box mAP50: {box_map50:.4f} | Mask mAP50 (Width Precision): {seg_map50:.4f}")

    print("\n" + "=" * 65)
    print("FINAL BENCHMARK WINNER")
    print("=" * 65)
    if results_summary:
        best_model = max(results_summary, key=lambda k: results_summary[k]["seg_map50"])
        for name, res in results_summary.items():
            marker = " 🏆 [WINNER FOR PRODUCTION]" if name == best_model else ""
            print(f"{name}: Mask mAP50 = {res['seg_map50']:.4f}{marker}")
    else:
        print("No valid model weights found to evaluate.")