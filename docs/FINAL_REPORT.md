
# 🛣️ Road Damage Detection - Final Project Report

## 📋 Evaluation Metrics
The model has been evaluated on the balanced validation set. Below are the quantitative performance metrics:

| Class | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
|-------|---------------|------------|-------|----------|
| **Overall** | 0.4376 | 0.3513 | 0.3248 | 0.1530 |
| Longitudinal Crack | 0.1435* | - | - | - |
| Transverse Crack | 0.1469* | - | - | - |
| Alligator Crack | 0.2246* | - | - | - |
| Pothole | 0.0972* | - | - | - |

*(Note: Class-wise map50 values are shown above)*

## ⏱️ Performance Speed
- **Pre-process**: 0.33 ms per image
- **Inference**: 23.37 ms per image
- **Post-process**: 1.40 ms per image

## 📂 Project Assets
- **Best Model Weights**: `models/road_damage_best.pt`
- **Training Source**: `data/images`
- **Report Location**: `docs/FINAL_REPORT.md`
