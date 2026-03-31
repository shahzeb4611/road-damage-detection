from ultralytics import YOLO
import os

def resume_training():
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Path to the last weights
    last_weights_path = os.path.join(project_root, 'runs', 'detect', 'rdd2022_full_run_cpu', 'weights', 'last.pt')
    
    # Load the model from the last checkpoint
    model = YOLO(last_weights_path)
    
    # Resume training
    print(f"Resuming training from: {last_weights_path}")
    results = model.train(resume=True)
    
    print("Training complete!")
    print(f"Model saved to: {results.save_dir}")

if __name__ == "__main__":
    resume_training()
