from ultralytics import YOLO
import cv2
import os

def run_webcam():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load our optimized best model
    model_path = os.path.join(project_root, 'models', 'road_damage_best.pt')
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    # Open the webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to stop the webcam feed.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference on the frame
        # stream=True for memory efficiency in loops
        results = model.predict(frame, conf=0.25, verbose=False)

        # Draw results on the frame
        annotated_frame = results[0].plot()

        # Display the frame
        cv2.imshow("Road Damage Detection - Live Feed", annotated_frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()
