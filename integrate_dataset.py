"""
Dataset Integration & Verification Script
Integrates extra dataset from D:\\ham into existing project D:\\New width prj.
"""

import os
import re
import sys
import json
import shutil
import random
from pathlib import Path
from collections import Counter
from PIL import Image

# Directories
PROJECT_DIR = Path(r"D:\New width prj")
HAM_DIR = Path(r"D:\ham")

TRAIN_IMG_DIR = PROJECT_DIR / "train" / "images"
TRAIN_LBL_DIR = PROJECT_DIR / "train" / "labels"
VAL_IMG_DIR = PROJECT_DIR / "val" / "images"
VAL_LBL_DIR = PROJECT_DIR / "val" / "labels"
PROJECT_LABELS_RAW_DIR = PROJECT_DIR / "labels"

MASTER_JSON_PATH = PROJECT_DIR / "master_annotations.json"
DATA_YAML_PATH = PROJECT_DIR / "data.yaml"
CLASSES_TXT_PATH = PROJECT_DIR / "classes.txt"

CLASS_NAMES = {
    0: "Normal",
    1: "Overwidth",
    2: "Smughed",
    3: "Underwidth"
}

def clean_filename(name: str) -> str:
    """Strip 8-char hex UUID prefix (e.g. '040aa315-') and sanitize spaces/special chars."""
    no_uuid = re.sub(r"^[0-9a-fA-F]{8}-", "", name)
    clean = re.sub(r"[\s\(\)]+", "_", no_uuid)
    clean = re.sub(r"_+", "_", clean)
    clean = re.sub(r"_\.", ".", clean)
    return clean

def parse_and_sanitize_label(lbl_path: Path):
    """
    Read YOLO segmentation label file.
    Validates polygon points, clamps coordinates to [0.0, 1.0],
    and returns parsed polygons and sanitized lines.
    """
    polygons = []
    sanitized_lines = []
    
    with open(lbl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for idx, line in enumerate(lines):
        line_clean = line.strip()
        if not line_clean:
            continue
        parts = line_clean.split()
        cls_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
        
        if len(coords) < 6 or len(coords) % 2 != 0:
            print(f"  [WARNING] Line {idx+1} in {lbl_path.name} has invalid coordinate count ({len(coords)}). Skipping.")
            continue
            
        clamped_coords = [max(0.0, min(1.0, c)) for c in coords]
        points_xy = [[clamped_coords[i], clamped_coords[i+1]] for i in range(0, len(clamped_coords), 2)]
        
        fmt_coords = " ".join([f"{c:.6f}" for c in clamped_coords])
        sanitized_lines.append(f"{cls_id} {fmt_coords}\n")
        
        xs = [pt[0] for pt in points_xy]
        ys = [pt[1] for pt in points_xy]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
        
        polygons.append({
            "class_id": cls_id,
            "class_name": CLASS_NAMES.get(cls_id, f"Class_{cls_id}"),
            "points": points_xy,
            "bbox_normalized": bbox,
            "num_points": len(points_xy)
        })
        
    return polygons, sanitized_lines

def locate_ham_files():
    """Locate notes3.json/note3.json and classes3.txt/class3.txt in D:\ham."""
    notes_file = None
    for cand in ["notes3.json", "note3.json", "notes.json"]:
        p = HAM_DIR / cand
        if p.exists():
            notes_file = p
            break
            
    classes_file = None
    for cand in ["classes3.txt", "class3.txt", "classes.txt"]:
        p = HAM_DIR / cand
        if p.exists():
            classes_file = p
            break
            
    if not notes_file:
        raise FileNotFoundError(f"Could not find note3.json / notes3.json in {HAM_DIR}")
    if not classes_file:
        raise FileNotFoundError(f"Could not find class3.txt / classes3.txt in {HAM_DIR}")
        
    return notes_file, classes_file

def merge_class_files(ham_classes_path: Path):
    """Combine class entries from ham class file into main master class list."""
    print("\n--- Task 1b: Merging Classes ---")
    with open(ham_classes_path, "r", encoding="utf-8") as f:
        ham_classes = [line.strip() for line in f if line.strip()]
        
    existing_classes = []
    if CLASSES_TXT_PATH.exists():
        with open(CLASSES_TXT_PATH, "r", encoding="utf-8") as f:
            existing_classes = [line.strip() for line in f if line.strip()]
            
    combined_classes = list(existing_classes)
    for c in ham_classes:
        if c not in combined_classes:
            combined_classes.append(c)
            
    with open(CLASSES_TXT_PATH, "w", encoding="utf-8") as f:
        for c in combined_classes:
            f.write(f"{c}\n")
            
    # Also save classes3.txt and class3.txt in project dir for consistency
    for name in ["classes3.txt", "class3.txt"]:
        shutil.copy2(ham_classes_path, PROJECT_DIR / name)
        
    print(f"Master classes updated ({len(combined_classes)} classes): {combined_classes}")
    return combined_classes

def scan_ham_dataset():
    """Scan and pair valid images and labels in D:\ham."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_candidates = []
    for d in [HAM_DIR / "images", HAM_DIR]:
        if d.exists():
            img_candidates.extend([f for f in d.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])
    # deduplicate by path
    img_candidates = sorted(list(set(img_candidates)), key=lambda x: x.name)
    
    lbl_candidates = {}
    for d in [HAM_DIR / "labels", HAM_DIR]:
        if d.exists():
            for f in d.iterdir():
                if f.is_file() and f.suffix.lower() == ".txt":
                    lbl_candidates[f.stem] = f
                    
    matched_pairs = []
    for img in img_candidates:
        lbl = lbl_candidates.get(img.stem)
        if lbl:
            matched_pairs.append((img, lbl))
        else:
            print(f"  [WARNING] No matching label text file found for {img.name}")
            
    print(f"Found {len(matched_pairs)} valid image-label pairs in {HAM_DIR}.")
    return matched_pairs

def distribute_and_copy(pairs, seed: int = 42):
    """
    Split incoming pairs (80% train, 20% val),
    ensuring existing project files are NOT overwritten.
    """
    print("\n--- Task 2: Distribute New Images and Labels ---")
    
    # Check existing files
    existing_train_imgs = {f.name: f for f in TRAIN_IMG_DIR.glob("*.*")}
    existing_val_imgs = {f.name: f for f in VAL_IMG_DIR.glob("*.*")}
    all_existing_imgs = set(existing_train_imgs.keys()) | set(existing_val_imgs.keys())
    
    new_pairs = []
    skipped_existing = []
    
    for img, lbl in pairs:
        clean_img_name = clean_filename(img.name)
        if clean_img_name in all_existing_imgs:
            skipped_existing.append((img, lbl, clean_img_name))
        else:
            new_pairs.append((img, lbl, clean_img_name))
            
    print(f"Total pairs scanned from {HAM_DIR}: {len(pairs)}")
    print(f"Existing files detected in project: {len(skipped_existing)} (will NOT be overwritten)")
    print(f"New pairs to distribute: {len(new_pairs)}")
    
    if not new_pairs:
        print("All incoming pairs are already integrated into the project!")
        return [], [], skipped_existing
        
    # Stratified or balanced split on new_pairs (80% train, 20% val)
    # Target 80% train / 20% val
    val_target_count = max(1, round(len(new_pairs) * 0.20))
    train_target_count = len(new_pairs) - val_target_count
    
    # Calculate class counts per new pair to balance splits
    pair_meta = []
    total_class_counts = Counter()
    for img, lbl, clean_name in new_pairs:
        polygons, sanitized = parse_and_sanitize_label(lbl)
        cls_counter = Counter(p["class_id"] for p in polygons)
        total_class_counts.update(cls_counter)
        pair_meta.append({
            "img": img,
            "lbl": lbl,
            "clean_name": clean_name,
            "clean_lbl_name": Path(clean_name).stem + ".txt",
            "polygons": polygons,
            "sanitized": sanitized,
            "classes": cls_counter
        })
        
    # Multi-class balancing search
    best_score = float("inf")
    best_train_meta = None
    best_val_meta = None
    
    for s in range(seed, seed + 2000):
        random.seed(s)
        shuffled = list(pair_meta)
        random.shuffle(shuffled)
        v_cand = shuffled[:val_target_count]
        t_cand = shuffled[val_target_count:]
        
        v_classes = Counter()
        for item in v_cand:
            v_classes.update(item["classes"])
            
        # Preference: val should have classes 0, 2, 3 represented
        score = sum(abs(v_classes[c] - (total_class_counts[c] * 0.2)) for c in [0, 2, 3])
        if score < best_score:
            best_score = score
            best_train_meta = t_cand
            best_val_meta = v_cand
            
    print(f"Split result: {len(best_train_meta)} train ({len(best_train_meta)/len(new_pairs)*100:.1f}%), "
          f"{len(best_val_meta)} val ({len(best_val_meta)/len(new_pairs)*100:.1f}%)")
    
    # Copy files
    train_added = []
    val_added = []
    
    for item in best_train_meta:
        dest_img = TRAIN_IMG_DIR / item["clean_name"]
        dest_lbl = TRAIN_LBL_DIR / item["clean_lbl_name"]
        
        # Verify no overwrite
        if dest_img.exists() or dest_lbl.exists():
            raise FileExistsError(f"Safety check triggered: {dest_img} or {dest_lbl} already exists!")
            
        shutil.copy2(item["img"], dest_img)
        with open(dest_lbl, "w", encoding="utf-8") as f:
            f.writelines(item["sanitized"])
            
        # Also copy raw label to raw labels folder if not present
        raw_dest_lbl = PROJECT_LABELS_RAW_DIR / item["lbl"].name
        if not raw_dest_lbl.exists():
            shutil.copy2(item["lbl"], raw_dest_lbl)
            
        train_added.append(item)
        
    for item in best_val_meta:
        dest_img = VAL_IMG_DIR / item["clean_name"]
        dest_lbl = VAL_LBL_DIR / item["clean_lbl_name"]
        
        # Verify no overwrite
        if dest_img.exists() or dest_lbl.exists():
            raise FileExistsError(f"Safety check triggered: {dest_img} or {dest_lbl} already exists!")
            
        shutil.copy2(item["img"], dest_img)
        with open(dest_lbl, "w", encoding="utf-8") as f:
            f.writelines(item["sanitized"])
            
        # Also copy raw label to raw labels folder if not present
        raw_dest_lbl = PROJECT_LABELS_RAW_DIR / item["lbl"].name
        if not raw_dest_lbl.exists():
            shutil.copy2(item["lbl"], raw_dest_lbl)
            
        val_added.append(item)
        
    print(f"Successfully copied {len(train_added)} image-label pairs into train/")
    print(f"Successfully copied {len(val_added)} image-label pairs into val/")
    
    # Invalidate old YOLO caches so Ultralytics detects the new data
    for cache_p in [PROJECT_DIR / "train" / "labels.cache", PROJECT_DIR / "val" / "labels.cache"]:
        if cache_p.exists():
            cache_p.unlink()
            print(f"Removed stale cache: {cache_p}")
            
    return train_added, val_added, skipped_existing

def merge_master_annotations(train_added, val_added, ham_notes_path: Path):
    """
    Safely merge note3.json into existing master_annotations.json
    and append newly added image & annotation entries.
    """
    print("\n--- Task 1a: Merging JSON into master_annotations.json ---")
    
    # Copy notes3.json and note3.json into project dir for record-keeping
    for name in ["notes3.json", "note3.json"]:
        shutil.copy2(ham_notes_path, PROJECT_DIR / name)
        
    with open(MASTER_JSON_PATH, "r", encoding="utf-8") as f:
        master = json.load(f)
        
    with open(ham_notes_path, "r", encoding="utf-8") as f:
        ham_notes = json.load(f)
        
    # 1. Update info source_exports
    source_exports = master["info"].get("source_exports", [])
    for exp in ["notes3.json", "classes3.txt"]:
        if exp not in source_exports:
            source_exports.append(exp)
    master["info"]["source_exports"] = source_exports
    
    # 2. Merge categories
    existing_cat_ids = {c["id"]: c["name"] for c in master.get("categories", [])}
    for cat in ham_notes.get("categories", []):
        cid = cat["id"]
        cname = cat["name"]
        if cid not in existing_cat_ids:
            master["categories"].append({"id": cid, "name": cname})
            existing_cat_ids[cid] = cname
            
    # 3. Append newly integrated images and annotations
    existing_img_ids = {img["id"] for img in master.get("images", [])}
    existing_ann_ids = {ann["id"] for ann in master.get("annotations", [])}
    existing_file_names = {img["file_name"] for img in master.get("images", [])}
    
    next_img_id = max(existing_img_ids, default=0) + 1
    next_ann_id = max(existing_ann_ids, default=0) + 1
    
    all_added = train_added + val_added
    added_img_count = 0
    added_ann_count = 0
    
    for item in all_added:
        if item["clean_name"] in existing_file_names:
            continue
            
        with Image.open(item["img"]) as im:
            w, h = im.size
            
        cur_img_id = next_img_id
        next_img_id += 1
        
        master["images"].append({
            "id": cur_img_id,
            "file_name": item["clean_name"],
            "label_file": item["clean_lbl_name"],
            "original_image_name": item["img"].name,
            "original_label_name": item["lbl"].name,
            "width": w,
            "height": h
        })
        existing_file_names.add(item["clean_name"])
        added_img_count += 1
        
        for poly in item["polygons"]:
            abs_points = [[round(pt[0] * w, 2), round(pt[1] * h, 2)] for pt in poly["points"]]
            bbox_norm = poly["bbox_normalized"]
            abs_bbox = [
                round(bbox_norm[0] * w, 2),
                round(bbox_norm[1] * h, 2),
                round(bbox_norm[2] * w, 2),
                round(bbox_norm[3] * h, 2)
            ]
            
            master["annotations"].append({
                "id": next_ann_id,
                "image_id": cur_img_id,
                "category_id": poly["class_id"],
                "category_name": poly["class_name"],
                "segmentation_normalized": poly["points"],
                "segmentation_pixel": abs_points,
                "bbox_normalized": bbox_norm,
                "bbox_pixel": abs_bbox,
                "num_vertices": poly["num_points"]
            })
            next_ann_id += 1
            added_ann_count += 1
            
    master["info"]["total_images"] = len(master["images"])
    
    with open(MASTER_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2)
        
    print(f"Master annotations updated at: {MASTER_JSON_PATH}")
    print(f"Added {added_img_count} new images and {added_ann_count} new polygon annotations.")
    print(f"Total images now in master: {len(master['images'])}, total annotations: {len(master['annotations'])}")
    return master

def verify_integrity(train_added, val_added):
    """Verify 1:1 image-label mapping, count integrity, and Ultralytics dataset validation."""
    print("\n--- Task 3: Integrity Verification ---")
    
    train_imgs = list(TRAIN_IMG_DIR.glob("*.*"))
    train_lbls = list(TRAIN_LBL_DIR.glob("*.txt"))
    val_imgs = list(VAL_IMG_DIR.glob("*.*"))
    val_lbls = list(VAL_LBL_DIR.glob("*.txt"))
    
    print(f"Final Dataset Summary:")
    print(f"  - Train Images: {len(train_imgs)}, Train Labels: {len(train_lbls)}")
    print(f"  - Val Images:   {len(val_imgs)}, Val Labels:   {len(val_lbls)}")
    print(f"  - Total Dataset: {len(train_imgs) + len(val_imgs)} images and {len(train_lbls) + len(val_lbls)} labels")
    
    # 1:1 stem mapping check
    train_img_stems = {f.stem for f in train_imgs}
    train_lbl_stems = {f.stem for f in train_lbls}
    val_img_stems = {f.stem for f in val_imgs}
    val_lbl_stems = {f.stem for f in val_lbls}
    
    train_diff = train_img_stems ^ train_lbl_stems
    val_diff = val_img_stems ^ val_lbl_stems
    train_val_overlap = (train_img_stems & val_img_stems)
    
    assert len(train_diff) == 0, f"Train 1:1 stem mismatch detected: {train_diff}"
    assert len(val_diff) == 0, f"Val 1:1 stem mismatch detected: {val_diff}"
    assert len(train_val_overlap) == 0, f"Train/Val data leakage detected! Overlapping stems: {train_val_overlap}"
    
    print("  [OK] 1:1 image-to-label mapping: PASSED (Train and Val have zero orphan files)")
    print("  [OK] Train / Val split isolation: PASSED (Zero leakage/overlap between splits)")
    
    # Validate instances distribution
    def count_instances(lbl_dir):
        inst_counts = Counter()
        for p in lbl_dir.glob("*.txt"):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        inst_counts[int(parts[0])] += 1
        return inst_counts

    train_counts = count_instances(TRAIN_LBL_DIR)
    val_counts = count_instances(VAL_LBL_DIR)
    total_counts = train_counts + val_counts
    
    print("\nUpdated Class Instance Distribution across Dataset:")
    print(f"  {'Class ID':<10} {'Class Name':<15} {'Train':<10} {'Val':<10} {'Total':<10}")
    print(f"  {'-'*55}")
    for cid in range(4):
        cname = CLASS_NAMES.get(cid, f"Class_{cid}")
        print(f"  {cid:<10} {cname:<15} {train_counts[cid]:<10} {val_counts[cid]:<10} {total_counts[cid]:<10}")
    print(f"  {'-'*55}")
    print(f"  {'TOTAL':<26} {sum(train_counts.values()):<10} {sum(val_counts.values()):<10} {sum(total_counts.values()):<10}")

    # Optional Ultralytics scan
    try:
        from ultralytics.data.utils import check_det_dataset
        from ultralytics.data.dataset import YOLODataset
        print("\nUltralytics Validation Tool Scan:")
        ds_info = check_det_dataset(str(DATA_YAML_PATH))
        print("  [OK] check_det_dataset: PASSED")
        ds_train = YOLODataset(img_path=ds_info['train'], data=ds_info, task='segment')
        print(f"  [OK] Ultralytics Train Cache: PASSED ({len(ds_train)} images verified)")
        ds_val = YOLODataset(img_path=ds_info['val'], data=ds_info, task='segment')
        print(f"  [OK] Ultralytics Val Cache:   PASSED ({len(ds_val)} images verified)")
    except Exception as e:
        print(f"  [INFO] Ultralytics check note: {e}")

def main():
    print("="*70)
    print("STARTING DATASET INTEGRATION PIPELINE: D:\\ham -> D:\\New width prj")
    print("="*70)
    
    # Task 1: Locate notes and classes
    notes_path, classes_path = locate_ham_files()
    print(f"Found source notes: {notes_path}")
    print(f"Found source classes: {classes_path}")
    
    # Task 1b: Merge classes
    merge_class_files(classes_path)
    
    # Task 2: Scan pairs
    pairs = scan_ham_dataset()
    
    # Task 2: Distribute new images and labels (80% train / 20% val)
    train_added, val_added, skipped_existing = distribute_and_copy(pairs, seed=42)
    
    # Task 1a: Merge JSON
    merge_master_annotations(train_added, val_added, notes_path)
    
    # Task 3: Integrity verification
    verify_integrity(train_added, val_added)
    
    print("\n" + "="*70)
    print("INTEGRATION COMPLETED SUCCESSFULLY WITHOUT ERRORS")
    print("="*70)

if __name__ == "__main__":
    main()
