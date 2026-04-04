from ultralytics import YOLO
import os
import shutil

def resume_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    models_dir = os.path.join(project_root, 'models')
    last_pt_path = os.path.join(project_root, 'runs', 'detect', 'rdd2022_retrain_hard', 'weights', 'last.pt')
    
    if not os.path.exists(last_pt_path):
        print(f"Error: {last_pt_path} not found. Cannot resume.")
        return
        
    print(f"🔄 Resuming training from {last_pt_path}...")
    model = YOLO(last_pt_path)
    
    # Resume training
    results = model.train(resume=True)
    
    # Move the best weight to models folder and remove last.pt
    save_dir = results.save_dir
    best_weight_path = os.path.join(save_dir, 'weights', 'best.pt')
    last_weight_path_new = os.path.join(save_dir, 'weights', 'last.pt')
    
    final_model_path = os.path.join(models_dir, 'road_damage_best.pt')
    
    if os.path.exists(best_weight_path):
        shutil.copy2(best_weight_path, final_model_path)
        print(f"✅ Best model saved to: {final_model_path}")
    
    if os.path.exists(last_weight_path_new):
        os.remove(last_weight_path_new)
        print("🗑️ Deleted 'last.pt' to keep only the best performing weight.")
        
    print("🎉 Training process complete!")

if __name__ == "__main__":
    resume_model()
