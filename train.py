from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n-seg.pt")

    model.train(
        data="data.yaml",
        imgsz=640,
        epochs=150,
        batch=16,          # Optimized for RTX 4060 8GB VRAM
        freeze=10,         # Preserve feature backbone for stable transfer
        lr0=0.001,

        # --- Physical Robot Feedrate (2400 mm/min) Tuned Hyperparameters ---
        scale=0.15,        # Low scale jitter: camera-to-bed standoff distance is fixed on gantry
        degrees=10.0,      # Low rotation: straight bead toolpaths stay primarily axis-aligned (±10°)
        fliplr=0.5,        # Horizontal flip enabled: symmetric for left/right extrusion passes
        flipud=0.0,        # Disabled: nozzle is always top-down relative to the camera mount
        mosaic=0.5,        # Reduced from 1.0 to 0.5: prevents artificial multi-angle seams on straight beads
        copy_paste=0.4,    # Boosted to 0.4: synthesizes extra 'Smughed' and 'Underwidth' defects
        perspective=0.0,   # SJCAM action camera is rigidly mounted (fixed pitch/roll)
        retina_masks=True, # Critical for sub-millimeter edge segmentation metrology

        workers=4,
        device=0,
        name="train-6",
    )