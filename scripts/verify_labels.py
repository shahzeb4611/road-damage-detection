import os
import cv2
import glob
import random
import matplotlib.pyplot as plt
from pathlib import Path

def verify_labels(num_samples=5):
    """
    Randomly selects images from the training dataset, reads their corresponding
    YOLOv8 format bounding boxes, draws them, and saves the verified images.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Path to images and labels
    images_dir = os.path.join(project_root, 'data_full', 'images', 'train')
    labels_dir = os.path.join(project_root, 'data_full', 'labels', 'train')
    
    # Output directory for verified samples
    output_dir = os.path.join(project_root, 'outputs', 'label_verification')
    os.makedirs(output_dir, exist_ok=True)
    
    # Class mapping based on full_data.yaml
    classes = {
        0: "longitudinal_crack",
        1: "transverse_crack",
        2: "alligator_crack",
        3: "pothole"
    }
    
    # Colors for each class (B, G, R)
    colors = {
        0: (255, 0, 0),    # Blue
        1: (0, 255, 0),    # Green
        2: (0, 0, 255),    # Red
        3: (0, 255, 255)   # Yellow
    }
    
    # Get all training images
    image_paths = glob.glob(os.path.join(images_dir, "*.jpg"))
    if not image_paths:
        print(f"No images found in {images_dir}")
        return
        
    # Sample random images
    sampled_images = random.sample(image_paths, min(num_samples, len(image_paths)))
    
    print(f"Verifying {len(sampled_images)} random training samples...")
    
    for img_path in sampled_images:
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue
            
        h, w, _ = img.shape
        img_name = os.path.basename(img_path)
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, txt_name)
        
        # Check if label exists
        has_labels = False
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    has_labels = True
                    class_id = int(parts[0])
                    x_center, y_center = float(parts[1]), float(parts[2])
                    width, height = float(parts[3]), float(parts[4])
                    
                    # Convert YOLO format back to pixel coordinates
                    x1 = int((x_center - width/2) * w)
                    y1 = int((y_center - height/2) * h)
                    x2 = int((x_center + width/2) * w)
                    y2 = int((y_center + height/2) * h)
                    
                    color = colors.get(class_id, (255, 255, 255))
                    label = classes.get(class_id, f"Class {class_id}")
                    
                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label text
                    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, (x1, y1 - text_h - baseline - 5), (x1 + text_w, y1), color, -1)
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
        # Save output
        output_path = os.path.join(output_dir, f"verified_{img_name}")
        cv2.imwrite(output_path, img)
        
        status = "with bounding boxes" if has_labels else "with NO bounding boxes"
        print(f"Saved {output_path} ({status})")

    print(f"\n✅ Label verification complete! Check the '{output_dir}' directory.")

if __name__ == "__main__":
    verify_labels()
