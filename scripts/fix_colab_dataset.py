import os
import shutil
import random
from pathlib import Path

def fix_dataset():
    # Paths
    project_root = Path(r"e:\BS-CS-07\Computer Vision\Assignmnet-1")
    data_full = project_root / "data_full"
    
    train_images = data_full / "images" / "train"
    train_labels = data_full / "labels" / "train"
    
    val_images_dir = data_full / "images" / "val"
    val_labels_dir = data_full / "labels" / "val"
    
    print("🧹 Cleaning up old invalid validation set (empty labels)...")
    if val_images_dir.exists():
        shutil.rmtree(val_images_dir)
    if val_labels_dir.exists():
        shutil.rmtree(val_labels_dir)
        
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # Get all training images
    all_train_images = [f for f in os.listdir(train_images) if f.endswith(('.jpg', '.png'))]
    print(f"Found {len(all_train_images)} total training images.")
    
    # We want ~10% for validation (about 3,800)
    val_size = int(len(all_train_images) * 0.10)
    
    print(f"🔀 Moving {val_size} images and labels to the new validation set...")
    
    # Randomly select images for validation
    random.seed(42) # For reproducibility
    val_images = random.sample(all_train_images, val_size)
    
    moved_count = 0
    for img_name in val_images:
        img_src = train_images / img_name
        img_dst = val_images_dir / img_name
        
        label_name = img_name.rsplit('.', 1)[0] + '.txt'
        label_src = train_labels / label_name
        label_dst = val_labels_dir / label_name
        
        # Move image
        if img_src.exists():
            shutil.move(str(img_src), str(img_dst))
        
        # Move label
        if label_src.exists():
            shutil.move(str(label_src), str(label_dst))
            
        moved_count += 1
        if moved_count % 500 == 0:
            print(f"  Moved {moved_count} / {val_size}...")
            
    print("\n✅ Dataset fix complete!")
    print(f"New Training Set: {len(os.listdir(train_images))} images")
    print(f"New Validation Set: {len(os.listdir(val_images_dir))} images")
    print("You can now zip the 'data_full' folder again and upload to Google Drive!")

if __name__ == "__main__":
    fix_dataset()
