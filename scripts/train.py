from ultralytics import YOLO
import os
import shutil
import glob

def train_model():
    # Load a model
    # Switching to yolov8s.pt (small) to greatly improve accuracy.
    model = YOLO('yolov8s.pt') 

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    models_dir = os.path.join(project_root, 'models')
    
    # Delete previous weights
    print("🧹 Cleaning up previous weights...")
    for old_weight in glob.glob(os.path.join(models_dir, '*.pt')):
        try:
            os.remove(old_weight)
            print(f"  - Deleted: {old_weight}")
        except PermissionError:
            print(f"  - Skipping {old_weight} (locked by another process)")
    
    # Path to balanced_data.yaml instead of full_data.yaml since it's already verified
    data_path = os.path.join(project_root, 'data', 'balanced_data.yaml')
    
    # Train the model
    # Increasing imgsz and epochs for a "harder", more accurate training.
    print("🚀 Starting intense training for high accuracy...")
    results = model.train(
        data=data_path,
        epochs=50, 
        imgsz=640, 
        batch=4, 
        name='rdd2022_retrain_hard',
        device='cpu' 
    )
    
    # Move the best weight to models folder and remove last.pt
    save_dir = results.save_dir
    best_weight_path = os.path.join(save_dir, 'weights', 'best.pt')
    last_weight_path = os.path.join(save_dir, 'weights', 'last.pt')
    
    final_model_path = os.path.join(models_dir, 'road_damage_best.pt')
    
    if os.path.exists(best_weight_path):
        shutil.copy2(best_weight_path, final_model_path)
        print(f"✅ Best model saved to: {final_model_path}")
    
    if os.path.exists(last_weight_path):
        os.remove(last_weight_path)
        print("🗑️ Deleted 'last.pt' to keep only the best performing weight.")
        
    print("🎉 Training process complete!")

if __name__ == "__main__":
    train_model()
