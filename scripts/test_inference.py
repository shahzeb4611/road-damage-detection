from ultralytics import YOLO
import os
import random

def run_inference():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load the best weights from our final training run
    model_path = os.path.join(project_root, 'models', 'best.pt')
    model = YOLO(model_path)

    # Path to sample images for testing
    test_img_dir = os.path.join(project_root, 'data', 'sample_images')
    
    if not os.path.exists(test_img_dir):
        print(f"Test directory not found: {test_img_dir}")
        return

    # Pick 10 random images from the test set
    test_files = [f for f in os.listdir(test_img_dir) if f.endswith('.jpg')]
    if not test_files:
        print("No test images found.")
        return
        
    sample_images = random.sample(test_files, min(10, len(test_files)))
    
    # Run prediction with a lower confidence to see "candidate" detections
    results = model.predict(
        source=[os.path.join(test_img_dir, img) for img in sample_images],
        save=True,
        conf=0.15, # Lower threshold to see what it's learning
        name='rdd_subset_test_results'
    )
    
    print("\nInference complete!")
    print(f"Predicted images saved to: {results[0].save_dir}")

if __name__ == "__main__":
    run_inference()
