import json
from pathlib import Path

# Paths
synth_labels_dir = Path(r"D:\width-measure\synthetic_concrete_geometry_v1\labels")
metadata_path = Path(r"D:\width-measure\synthetic_concrete_geometry_v1\generation_metadata\dataset_metadata.json")

# Load metadata list directly
print(f"Loading metadata from: {metadata_path}")
with open(metadata_path, 'r') as f:
    metadata_list = json.load(f)

# Build a fast lookup: filename_stem -> nominal_width_mm using 'image_rel_path'
width_lookup = {}
for entry in metadata_list:
    # Example image_rel_path: "train/images/rounded_90_turns_0000.jpg" -> get stem
    rel_path = entry.get("image_rel_path", "")
    if not rel_path:
        continue
    stem = Path(rel_path).stem
    width_lookup[stem] = entry["nominal_width_mm"]

print(f"Loaded {len(width_lookup)} width definitions from metadata.")

# Process all split label folders (train, val, test)
splits = ['train', 'val', 'test']
for split in splits:
    split_dir = synth_labels_dir / split
    if not split_dir.exists():
        print(f"Skipping {split} (folder not found).")
        continue
        
    txt_files = list(split_dir.glob("*.txt"))
    print(f"Processing {split} split: {len(txt_files)} label files...")
    
    updated_count = 0
    for txt_file in txt_files:
        stem = txt_file.stem
        if stem not in width_lookup:
            continue
            
        nominal_width = width_lookup[stem]
        
        # Determine correct class_id based on your metrology specs
        # 0: Normal (40-60mm), 1: Overwidth (>60mm), 3: Underwidth (<40mm)
        if nominal_width < 40.0:
            new_class_id = 3  # Underwidth
        elif nominal_width > 60.0:
            new_class_id = 1  # Overwidth
        else:
            new_class_id = 0  # Normal
            
        # Read lines, replace first token (class_id)
        lines = txt_file.read_text().splitlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            parts[0] = str(new_class_id)
            new_lines.append(" ".join(parts))
            
        # Overwrite file with corrected class IDs
        txt_file.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
        updated_count += 1

    print(f"Successfully remapped {updated_count} files in {split}.")

print("\nAll synthetic labels have been successfully remapped to match your 4-class project schema!")