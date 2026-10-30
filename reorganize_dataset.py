import os
import re
import shutil
import random
from pathlib import Path

def prepare_yolo_dataset(root_dir: str, val_split_ratio: float = 0.2, seed: int = 42):
    root = Path(root_dir)
    print(f"=== Starting YOLO Dataset Organization for: {root} ===")

    # 1. Source directories (Original clean sources)
    raw_images_dir = root / "images"
    raw_labels_dir = root / "labels"
    train_images_dir = root / "train" / "images"
    train_labels_dir = root / "train" / "labels"
    val_images_dir = root / "val" / "images"
    val_labels_dir = root / "val" / "labels"

    # 2. Ensure standard YOLO subdirectories exist
    for d in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        d.mkdir(parents=True, exist_ok=True)
    print("Standard subdirectories verified.")

    # 3. Read image files from the source root/images
    raw_imgs = [f for f in raw_images_dir.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
    raw_lbls = [f for f in raw_labels_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"]

    print(f"Source images in 'images/': {len(raw_imgs)}")
    print(f"Source labels in 'labels/': {len(raw_lbls)}")

    # 4. Match image and label files
    matched_pairs = []
    unmatched_imgs = []

    for img_path in raw_imgs:
        # Standardize image filename: replace spaces with underscores, drop parentheses
        target_img_name = img_path.name.replace(" ", "_").replace("(", "").replace(")", "")
        img_norm = re.sub(r"[^a-zA-Z0-9]", "", img_path.stem).lower()
        
        matched_lbl = None
        for lbl_path in raw_lbls:
            # Strip UUID prefix if present: e.g. 1063280e-CAM_C_Single...
            clean_lbl_stem = re.sub(r"^[0-9a-fA-F]{8}-", "", lbl_path.stem)
            lbl_norm = re.sub(r"[^a-zA-Z0-9]", "", clean_lbl_stem).lower()
            if img_norm == lbl_norm:
                matched_lbl = lbl_path
                break
                
        if matched_lbl:
            target_lbl_name = Path(target_img_name).stem + ".txt"
            matched_pairs.append((target_img_name, img_path, target_lbl_name, matched_lbl))
        else:
            unmatched_imgs.append(img_path)

    if unmatched_imgs:
        raise RuntimeError(f"Error: {len(unmatched_imgs)} images could not be paired: {unmatched_imgs}")

    print(f"Successfully matched all {len(matched_pairs)} image-label pairs.")

    # 5. Clear target train/ and val/ folders cleanly before copying
    for d in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        for item in d.iterdir():
            if item.is_file():
                item.unlink()

    # 6. Split dataset (Train ~80%, Val ~20%)
    random.seed(seed)
    random.shuffle(matched_pairs)

    total_pairs = len(matched_pairs)
    val_count = max(1, round(total_pairs * val_split_ratio))
    val_pairs = matched_pairs[:val_count]
    train_pairs = matched_pairs[val_count:]

    print(f"Splitting: {len(train_pairs)} pairs to train ({100 - (val_count/total_pairs)*100:.1f}%), "
          f"{len(val_pairs)} pairs to val ({(val_count/total_pairs)*100:.1f}%)")

    # Helper function to sanitize coordinate bounds when copying labels
    def copy_and_sanitize_label(src_path: Path, dest_path: Path):
        with open(src_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        clean_lines = []
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split()
            cls_id = parts[0]
            # Clip coords to [0.0, 1.0]
            sanitized_coords = []
            for c in parts[1:]:
                val = float(c)
                clipped = max(0.0, min(1.0, val))
                sanitized_coords.append(f"{clipped:.6f}")
            clean_lines.append(f"{cls_id} " + " ".join(sanitized_coords) + "\n")
        with open(dest_path, "w", encoding="utf-8") as f:
            f.writelines(clean_lines)

    # Populate train subdirectories
    for img_name, img_src, lbl_name, lbl_src in train_pairs:
        dest_img = train_images_dir / img_name
        dest_lbl = train_labels_dir / lbl_name
        shutil.copy2(img_src, dest_img)
        copy_and_sanitize_label(lbl_src, dest_lbl)

    # Populate val subdirectories
    for img_name, img_src, lbl_name, lbl_src in val_pairs:
        dest_img = val_images_dir / img_name
        dest_lbl = val_labels_dir / lbl_name
        shutil.copy2(img_src, dest_img)
        copy_and_sanitize_label(lbl_src, dest_lbl)

    # 7. Verification and summary report
    print("\n" + "="*50)
    print("VERIFICATION & SUMMARY REPORT")
    print("="*50)

    train_imgs = list(train_images_dir.glob("*.*"))
    train_lbls = list(train_labels_dir.glob("*.txt"))
    val_imgs = list(val_images_dir.glob("*.*"))
    val_lbls = list(val_labels_dir.glob("*.txt"))

    print(f"Train set: {len(train_imgs)} images, {len(train_lbls)} labels")
    print(f"Val set:   {len(val_imgs)} images, {len(val_lbls)} labels")

    # Verify exact stem pairing in train
    train_mismatches = []
    for img in train_imgs:
        expected_lbl = train_labels_dir / f"{img.stem}.txt"
        if not expected_lbl.exists():
            train_mismatches.append(img.name)

    # Verify exact stem pairing in val
    val_mismatches = []
    for img in val_imgs:
        expected_lbl = val_labels_dir / f"{img.stem}.txt"
        if not expected_lbl.exists():
            val_mismatches.append(img.name)

    print(f"Train stem mismatch count: {len(train_mismatches)}")
    print(f"Val stem mismatch count:   {len(val_mismatches)}")

    if train_mismatches or val_mismatches:
        print("ERROR: Mismatched files detected!")
        return False

    print("\nSUCCESS: All images and labels are 100% synchronized and match YOLOv11 standards!")
    return True

if __name__ == "__main__":
    prepare_yolo_dataset(r"D:\New width prj", val_split_ratio=0.2, seed=42)
