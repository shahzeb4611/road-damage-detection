from ultralytics import YOLO
import os

def train_model():
    # Load a model
    # We use yolov8n.pt (nano) as a starting point for efficiency, 
    # but you can use yolov8s.pt (small) or yolov8m.pt (medium) for better accuracy.
    model = YOLO('yolov8n.pt') 

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Path to full_data.yaml
    data_path = os.path.join(project_root, 'data_full', 'full_data.yaml')
    
    # Train the model
    # imgsz=320 is good for CPU, 
    # train on CPU as requested
    results = model.train(
        data=data_path,
        epochs=10, 
        imgsz=320, 
        batch=8, 
        name='rdd2022_full_run_cpu',
        device='cpu' 
    )
    
    print("Training complete!")
    print(f"Model saved to: {results.save_dir}")

if __name__ == "__main__":
    train_model()
