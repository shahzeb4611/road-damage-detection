import os
import glob
import random
import cv2
import shutil
from ultralytics import YOLO

def evaluate_and_visualize():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load the best model
    model_path = os.path.join(project_root, 'models', 'road_damage_best.pt')
    if not os.path.exists(model_path):
        # Fallback to yolov8n.pt if best.pt is not found for this demonstration
        print(f"Model {model_path} not found. Using baseline yolov8n.pt")
        model_path = os.path.join(project_root, 'yolov8n.pt')
        
    model = YOLO(model_path)
    
    # Output directories
    output_dir = os.path.join(project_root, 'outputs', 'evaluation')
    sample_dir = os.path.join(output_dir, 'sample_outputs')
    failure_dir = os.path.join(output_dir, 'failure_cases')
    
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(failure_dir, exist_ok=True)
    
    # Val data path
    val_images_dir = os.path.join(project_root, 'data_full', 'images', 'val')
    val_labels_dir = os.path.join(project_root, 'data_full', 'labels', 'val')
    
    images = glob.glob(os.path.join(val_images_dir, "*.jpg"))
    if not images:
        print("No validation images found.")
        return
        
    # --- 1. Generate Sample Output Visualizations ---
    print("Generating sample output visualizations...")
    sample_images = random.sample(images, min(5, len(images)))
    
    for img_path in sample_images:
        img_name = os.path.basename(img_path)
        # Run inference
        results = model.predict(source=img_path, save=False, conf=0.25)
        
        # Plot predictions
        res_plot = results[0].plot()
        out_path = os.path.join(sample_dir, f"pred_{img_name}")
        cv2.imwrite(out_path, res_plot)
        print(f"Saved sample prediction: {out_path}")
        
    # --- 2. Extract Validation Plots (Confusion Matrix) ---
    print("Extracting confusion matrix and PR curves...")
    # Assuming standard Ultralytics structure where validation runs are saved 
    # Try to find the latest val run
    runs_val_dir = os.path.join(project_root, 'runs', 'detect')
    if os.path.exists(runs_val_dir):
        val_runs = [os.path.join(runs_val_dir, d) for d in os.listdir(runs_val_dir) if d.startswith('val')]
        if val_runs:
            latest_val = sorted(val_runs, key=os.path.getmtime)[-1]
            conf_matrix_path = os.path.join(latest_val, 'confusion_matrix.png')
            if os.path.exists(conf_matrix_path):
                shutil.copy(conf_matrix_path, os.path.join(output_dir, 'confusion_matrix.png'))
                print("Copied confusion matrix.")
    
    # --- 3. Find Failure Cases ---
    print("Finding failure cases (False Positives / False Negatives)...")
    # This is a simplified heuristic: 
    # If the number of Ground Truths (GT) != number of Predictions (Preds), we treat it as a failure.
    # In reality, IoU calculations should be used, but this is a solid heuristic for this assignment.
    
    failure_count = 0
    target_failures = 5
    
    # Shuffle all images to randomly hunt for failures
    random.shuffle(images)
    
    for img_path in images:
        if failure_count >= target_failures:
            break
            
        img_name = os.path.basename(img_path)
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(val_labels_dir, txt_name)
        
        num_gt = 0
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                num_gt = len(f.readlines())
                
        # Run inference
        results = model.predict(source=img_path, verbose=False, conf=0.25)
        num_preds = len(results[0].boxes)
        
        # Failure detected: Length mismatch
        if num_gt != num_preds:
            # We want to save both the GT image and the Predicted image for comparison
            # Plot predictions
            pred_plot = results[0].plot()
            
            # Load original image, we'll annotate it with GT as well
            img_gt = cv2.imread(img_path)
            h, w, _ = img_gt.shape
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center = float(parts[1]), float(parts[2])
                            width, height = float(parts[3]), float(parts[4])
                            
                            x1 = int((x_center - width/2) * w)
                            y1 = int((y_center - height/2) * h)
                            x2 = int((x_center + width/2) * w)
                            y2 = int((y_center + height/2) * h)
                            
                            # GT drawn in distinctly fat green color
                            cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(img_gt, f"GT:{class_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
            # Combine side-by-side: Left = Ground Truth, Right = Prediction
            target_size = (640, min(640, int(640 * h / w))) # Resize for side-by-side
            img_gt_resized = cv2.resize(img_gt, target_size)
            pred_plot_resized = cv2.resize(pred_plot, target_size)
            
            combined = cv2.hconcat([img_gt_resized, pred_plot_resized])
            
            out_failure_path = os.path.join(failure_dir, f"failure_{img_name}")
            cv2.imwrite(out_failure_path, combined)
            
            print(f"Saved failure case {failure_count+1}: {img_name} (GT: {num_gt}, Preds: {num_preds})")
            failure_count += 1
            
    print("\nEvaluation visualization complete!")

if __name__ == "__main__":
    evaluate_and_visualize()
