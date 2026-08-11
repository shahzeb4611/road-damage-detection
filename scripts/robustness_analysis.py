import os
import cv2
import numpy as np
import random
from ultralytics import YOLO

def apply_poor_lighting(img):
    """Simulate low light by decreasing brightness and contrast."""
    # Convert to float, scale, and clip
    low_light = img.astype(float) * 0.4
    return np.clip(low_light, 0, 255).astype(np.uint8)

def apply_motion_blur(img, kernel_size=15):
    """Simulate motion blur using a diagonal kernel."""
    kernel = np.zeros((kernel_size, kernel_size))
    # Fill diagonal
    for i in range(kernel_size):
        kernel[i, i] = 1.0
    kernel = kernel / kernel_size
    return cv2.filter2D(img, -1, kernel)

def apply_occlusion(img):
    """Simulate occlusion by drawing random black rectangles."""
    h, w, _ = img.shape
    occluded = img.copy()
    num_blocks = random.randint(3, 6)
    for _ in range(num_blocks):
        bw = random.randint(40, 80)
        bh = random.randint(40, 80)
        x1 = random.randint(0, w - bw)
        y1 = random.randint(0, h - bh)
        # Draw dark gray rectangle (e.g. simulated debris/patch)
        cv2.rectangle(occluded, (x1, y1), (x1+bw, y1+bh), (30, 30, 30), -1)
    return occluded

def apply_gaussian_noise(img):
    """Simulate sensor noise/unseen data degradation."""
    row, col, ch = img.shape
    mean = 0
    var = 400
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = img.astype(float) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def analyze_robustness():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load fine-tuned model
    model_path = os.path.join(project_root, 'models', 'road_damage_best.pt')
    if not os.path.exists(model_path):
        print(f"❌ Custom model not found at {model_path}")
        return
        
    model = YOLO(model_path)
    
    # Directories
    img_dir = os.path.join(project_root, 'data', 'sample_images')
    output_dir = os.path.join(project_root, 'outputs', 'robustness')
    os.makedirs(output_dir, exist_ok=True)
    
    images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    if not images:
        print("❌ No sample images found.")
        return
        
    print(f"🔍 Analyzing robustness on {len(images)} sample images...")
    
    stats = {
        'Original': 0,
        'Poor Lighting': 0,
        'Motion Blur': 0,
        'Occlusion': 0,
        'Sensor Noise': 0
    }
    
    for img_name in images:
        path = os.path.join(img_dir, img_name)
        img = cv2.imread(path)
        if img is None:
            continue
            
        # Run original inference
        res_orig = model.predict(img, verbose=False, conf=0.20)
        stats['Original'] += len(res_orig[0].boxes)
        
        # Apply and run perturbations
        perturbations = {
            'Poor Lighting': apply_poor_lighting(img),
            'Motion Blur': apply_motion_blur(img),
            'Occlusion': apply_occlusion(img),
            'Sensor Noise': apply_gaussian_noise(img)
        }
        
        # Create a combined visualization sheet for each image
        # Size: original size for layout
        h, w, _ = img.shape
        grid_h = h // 2
        grid_w = w // 2
        
        # Plot original prediction
        plot_orig = cv2.resize(res_orig[0].plot(), (grid_w, grid_h))
        
        plots = [plot_orig]
        
        for name, p_img in perturbations.items():
            res_p = model.predict(p_img, verbose=False, conf=0.20)
            stats[name] += len(res_p[0].boxes)
            
            # Save plotted prediction
            plot_p = cv2.resize(res_p[0].plot(), (grid_w, grid_h))
            plots.append(plot_p)
            
            # Save individual perturbed images for reference
            cv2.imwrite(os.path.join(output_dir, f"{name.lower().replace(' ', '_')}_{img_name}"), res_p[0].plot())
            
        # Make a 2x3 combined collage (Original + 4 perturbations + 1 blank/legend info)
        # 3 columns, 2 rows
        row1 = cv2.hconcat([plots[0], plots[1], plots[2]])
        # Create a legend/info block for the 6th slot
        info_block = np.zeros((grid_h, grid_w, 3), dtype=np.uint8) + 40
        cv2.putText(info_block, "Robustness Test Sheet", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(info_block, f"Img: {img_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(info_block, f"Orig Dets: {len(res_orig[0].boxes)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        row2 = cv2.hconcat([plots[3], plots[4], info_block])
        collage = cv2.vconcat([row1, row2])
        
        cv2.imwrite(os.path.join(output_dir, f"collage_{img_name}"), collage)
        
    print("\n================ ROBUSTNESS REPORT ================")
    print(f"Total detections across {len(images)} test cases:")
    for key, val in stats.items():
        pct = (val / stats['Original'] * 100) if stats['Original'] > 0 else 0
        print(f" - {key:15}: {val:3d} detections ({pct:.1f}% of original)")
    print("===================================================\n")
    print(f"Results and visual collages saved to: {output_dir}")

if __name__ == '__main__':
    analyze_robustness()
