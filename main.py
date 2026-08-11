"""
Main entry point for Road Damage Detection System.

This script provides a unified command-line interface to train, evaluate, run predictions,
perform robustness analysis, and launch the web dashboard.
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path

# Add project root to python path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

def run_dashboard():
    """Launch the Streamlit web dashboard."""
    print("\n[Dashboard] Launching Road Damage Detection Web Dashboard...\n")
    app_path = os.path.join(project_root, "app.py")
    
    # Try using virtual environment python if available
    venv_python = os.path.join(project_root, "venv", "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        venv_python = os.path.join(project_root, "venv", "bin", "python") # Unix fallback
        
    cmd = [venv_python if os.path.exists(venv_python) else "python", "-m", "streamlit", "run", app_path]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[Dashboard] Dashboard stopped.")
    except Exception as e:
        print(f"[Error] Error starting Streamlit: {e}")

def run_train():
    """Run model training/fine-tuning."""
    print("\n[Train] Starting model training on the balanced dataset...\n")
    from scripts.train import train_model
    train_model()

def run_evaluate():
    """Run model evaluation and generate failure case visualizations."""
    print("\n[Evaluate] Running model evaluation and generating validation plots...\n")
    from scripts.evaluate_and_visualize import evaluate_and_visualize
    evaluate_and_visualize()

def run_predict(images=None, conf=0.25):
    """Run model prediction on sample or specified images."""
    print("\n[Predict] Running predictions...")
    from ultralytics import YOLO
    
    model_path = os.path.join(project_root, "models", "road_damage_best.pt")
    if not os.path.exists(model_path):
        print(f"[Error] Model not found at {model_path}.")
        return
        
    model = YOLO(model_path)
    
    if not images:
        # Default to sample images
        sample_dir = os.path.join(project_root, "data", "sample_images")
        if os.path.exists(sample_dir):
            images = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
            print(f"No images specified. Using {len(images)} sample images from: {sample_dir}")
        else:
            print("[Error] No images specified and sample images directory not found.")
            return

    results = model.predict(source=images, conf=conf, save=True, name="rdd_cli_predictions")
    print(f"[Success] Prediction complete. Results saved to: {results[0].save_dir}")

def run_robustness():
    """Run the robustness and edge case analysis."""
    print("\n[Robustness] Starting Robustness and Edge Case Analysis...\n")
    from scripts.robustness_analysis import analyze_robustness
    analyze_robustness()

def main():
    """Main CLI parser."""
    parser = argparse.ArgumentParser(
        description="Road Damage Detection System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch the web dashboard
  python main.py dashboard
  
  # Train model
  python main.py train
  
  # Evaluate model on validation set
  python main.py evaluate
  
  # Run prediction on sample images
  python main.py predict
  
  # Run prediction on specific images
  python main.py predict --images path/to/image1.jpg path/to/image2.jpg --conf 0.3
  
  # Run robustness analysis
  python main.py robustness
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Subcommands
    subparsers.add_parser("dashboard", help="Launch the Streamlit web dashboard")
    subparsers.add_parser("train", help="Train/fine-tune the model")
    subparsers.add_parser("evaluate", help="Evaluate the model and visualize results")
    subparsers.add_parser("robustness", help="Run simulated robustness/edge case analysis")
    
    predict_parser = subparsers.add_parser("predict", help="Run inference on images")
    predict_parser.add_argument("--images", nargs="+", help="Paths to specific images", default=None)
    predict_parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    
    args = parser.parse_args()
    
    if args.command == "dashboard":
        run_dashboard()
    elif args.command == "train":
        run_train()
    elif args.command == "evaluate":
        run_evaluate()
    elif args.command == "predict":
        run_predict(images=args.images, conf=args.conf)
    elif args.command == "robustness":
        run_robustness()
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
