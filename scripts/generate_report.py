from ultralytics import YOLO
import pandas as pd
import os

def generate_report():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load the best model
    model_path = os.path.join(project_root, 'models', 'best.pt')
    model = YOLO(model_path)

    # Path to your balanced_data.yaml
    data_path = os.path.join(project_root, 'data', 'balanced_data.yaml')

    print("Running validation for report...")
    # Validate the model
    results = model.val(data=data_path)
    
    # Extract metrics
    # results.results_dict contains mapping of metrics
    # results.speed contains time metrics
    
    # Create the report
    report = f"""
# 🛣️ Road Damage Detection - Final Project Report

## 📋 Evaluation Metrics
The model has been evaluated on the balanced validation set. Below are the quantitative performance metrics:

| Class | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
|-------|---------------|------------|-------|----------|
| **Overall** | {results.results_dict['metrics/precision(B)']:.4f} | {results.results_dict['metrics/recall(B)']:.4f} | {results.results_dict['metrics/mAP50(B)']:.4f} | {results.results_dict['metrics/mAP50-95(B)']:.4f} |
| Longitudinal Crack | {results.maps[0]:.4f}* | - | - | - |
| Transverse Crack | {results.maps[1]:.4f}* | - | - | - |
| Alligator Crack | {results.maps[2]:.4f}* | - | - | - |
| Pothole | {results.maps[3]:.4f}* | - | - | - |

*(Note: Class-wise map50 values are shown above)*

## ⏱️ Performance Speed
- **Pre-process**: {results.speed['preprocess']:.2f} ms per image
- **Inference**: {results.speed['inference']:.2f} ms per image
- **Post-process**: {results.speed['postprocess']:.2f} ms per image

## 📂 Project Assets
- **Best Model Weights**: `models/road_damage_best.pt`
- **Training Source**: `data/images`
- **Report Location**: `docs/FINAL_REPORT.md`
"""

    report_path = os.path.join(project_root, 'docs', 'FINAL_REPORT.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
        
    print(f"\n✅ Final report generated at: {report_path}")

if __name__ == "__main__":
    generate_report()
