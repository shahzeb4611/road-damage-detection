import os
import json
import shutil
from pathlib import Path

def convert():
    source_dir = Path(r"e:\BS-CS-07\Computer Vision\Assignmnet-1\rdd2022-DatasetNinja")
    dest_dir = Path(r"e:\BS-CS-07\Computer Vision\Assignmnet-1\data_full")
    
    # Class mapping
    class_map = {
        "longitudinal crack": 0,
        "transverse crack": 1,
        "alligator crack": 2,
        "pothole": 3
    }
    
    # Setup directories
    for split in ["train", "val"]:
        os.makedirs(dest_dir / "images" / split, exist_ok=True)
        os.makedirs(dest_dir / "labels" / split, exist_ok=True)
        
    for src_split, dst_split in [("train", "train"), ("test", "val")]:
        ann_dir = source_dir / src_split / "ann"
        img_dir = source_dir / src_split / "img"
        
        if not ann_dir.exists() or not img_dir.exists():
            continue
            
        print(f"Converting {src_split}...")
        
        # Count for progress
        count = 0
        for ann_file in ann_dir.glob("*.json"):
            with open(ann_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            img_width = data['size']['width']
            img_height = data['size']['height']
            
            # The annotation filename is typically like "img_name.json"
            # Get original image name by removing the ".json" extension
            img_name = ann_file.name[:-5]
            
            labels = []
            for obj in data.get('objects', []):
                class_title = obj.get('classTitle', '').lower()
                if class_title in class_map:
                    cls_id = class_map[class_title]
                    pts = obj.get('points', {}).get('exterior', [])
                    if len(pts) >= 2:
                        x1, y1 = pts[0]
                        x2, y2 = pts[1]
                        
                        # Calculate YOLO format values
                        x_center = (x1 + x2) / 2.0 / img_width
                        y_center = (y1 + y2) / 2.0 / img_height
                        w = abs(x2 - x1) / float(img_width)
                        h = abs(y2 - y1) / float(img_height)
                        
                        labels.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            
            # Save label file (only if there are objects or we can save empty files too, YOLO handles empty labels)
            # We must link the image regardless of label presence if it's meant to be in the dataset, but usually empty labels are fine.
            label_file = dest_dir / "labels" / dst_split / f"{Path(img_name).stem}.txt"
            with open(label_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(labels))
                
            # Create hardlink for image (or copy if hardlink fails)
            src_img = img_dir / img_name
            dst_img = dest_dir / "images" / dst_split / img_name
            
            if src_img.exists() and not dst_img.exists():
                try:
                    os.link(src_img, dst_img)
                except OSError:
                    # Fallback to copy
                    shutil.copy2(src_img, dst_img)
                    
            count += 1
            if count % 2000 == 0:
                print(f"Processed {count} images in {src_split}")

    # Create full_data.yaml
    yaml_content = f"""path: {dest_dir.absolute().as_posix()}
train: images/train
val: images/val

names:
  0: longitudinal_crack
  1: transverse_crack
  2: alligator_crack
  3: pothole
"""
    with open(dest_dir / "full_data.yaml", 'w') as f:
        f.write(yaml_content)
        
    print("Dataset conversion completed.")

if __name__ == "__main__":
    convert()
