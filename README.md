# 🛣️ Road Damage Detection System

An AI-powered road damage detection system built with **YOLOv8** and **Streamlit**. It detects four types of road damage — **longitudinal cracks, transverse cracks, alligator cracks, and potholes** — from images, videos, and live webcam feeds.

---

## 📸 Demo

| Image Detection | Video Processing |
|---|---|
| Upload one or multiple road images and get bounding boxes with confidence scores | Upload road videos; every frame is annotated. Pause anywhere to read crack type labels |

---

## 📁 Project Structure

```
road-damage-detection/
│
├── app.py                          # ✅ Main Streamlit web application
├── main.py                         # CLI entry point (train / evaluate / predict)
├── requirements.txt                # Python dependencies
├── .gitignore                      # Files excluded from Git
│
├── models/
│   └── road_damage_best.pt         # Trained YOLOv8 model weights ⬅️ (see below)
│
├── data/
│   ├── balanced_data.yaml          # Dataset config (balanced split)
│   ├── images/
│   │   ├── train/                  # Training images
│   │   └── val/                    # Validation images
│   ├── labels/
│   │   ├── train/                  # YOLO-format label files
│   │   └── val/
│   └── sample_images/              # Quick-test sample images
│
├── data_full/
│   ├── full_data.yaml              # Dataset config (full dataset)
│   ├── images/
│   └── labels/
│
├── scripts/
│   ├── train.py                    # Model training script
│   ├── evaluate_and_visualize.py   # Evaluation + plots
│   ├── test_inference.py           # Quick inference test
│   ├── verify_labels.py            # Dataset label verification
│   ├── convert_dataset.py          # Dataset format conversion
│   ├── resume_train.py             # Resume interrupted training
│   ├── webcam_inference.py         # Standalone webcam inference
│   ├── fix_colab_dataset.py        # Google Colab dataset helper
│   └── generate_report.py          # Auto-generate evaluation report
│
├── outputs/
│   ├── evaluation/
│   │   ├── confusion_matrix.png    # Model confusion matrix
│   │   ├── sample_outputs/         # Predicted images with bounding boxes
│   │   └── failure_cases/          # Images where model failed
│   └── label_verification/         # Verified label overlay images
│
└── docs/
    ├── ASSIGNMENT_1_REPORT.md      # Assignment 1 final report
    ├── Assignment_2_Report.md      # Assignment 2 report
    ├── FINAL_REPORT.md             # Project final report
    └── colab_training_guide.md     # Guide for training on Google Colab
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/road-damage-detection.git
cd road-damage-detection
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Trained Model

> ⚠️ The model file (`road_damage_best.pt`) is **not included in this repo** because it is ~6 MB.
> Download it from the link below and place it in the `models/` folder.

🔗 **[Download road_damage_best.pt – Google Drive](https://drive.google.com/YOUR_LINK_HERE)**

```
models/
└── road_damage_best.pt   ← place it here
```

### 5. Launch the Web App

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501** 🎉

---

## 🎯 Features

| Feature | Description |
|---|---|
| 🖼️ **Image Detection** | Upload one or multiple road images for batch damage detection |
| 🎬 **Video Processing** | Upload road videos — every frame gets annotated with crack overlays |
| 📷 **Live Webcam** | Real-time detection from your webcam feed |
| 🎚️ **Confidence Slider** | Tune detection sensitivity with live confidence threshold |
| 🔎 **Class Filter** | Show/hide specific damage types (longitudinal, transverse, alligator, pothole) |
| ⬇️ **Download Results** | Download annotated images and processed videos |

---

## 🏷️ Damage Classes

| Class | Description |
|---|---|
| 🔴 **Longitudinal Crack** | Crack parallel to the direction of traffic |
| 🟠 **Transverse Crack** | Crack perpendicular to the direction of traffic |
| 🟡 **Alligator Crack** | Interconnected web of cracks resembling alligator skin |
| 🟣 **Pothole** | Depression caused by erosion and wear under traffic load |

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Base Model | YOLOv8n (nano) |
| Dataset | RDD2022 (Road Damage Dataset 2022) |
| Training Epochs | 10 |
| Image Size | 320 × 320 |
| Device | CPU |
| Framework | Ultralytics YOLOv8 |

---

## 📊 Dataset

The model was trained on the **[RDD2022 dataset](https://github.com/sekilab/RoadDamageDetector)** which contains road images from multiple countries (Japan, India, Czech Republic, Norway, United States, China) annotated with road damage bounding boxes.

> 📂 Due to its large size, the dataset is **not included** in this repository.
> Download it from the official RDD2022 source and place images/labels into the `data/` or `data_full/` folders.

---

## 🔧 Retrain the Model (Optional)

If you want to train from scratch:

```bash
# Make sure data/balanced_data.yaml points to your dataset
python scripts/train.py

# Or use the CLI entry point
python main.py train
```

To resume interrupted training:

```bash
python scripts/resume_train.py
```

---

## 🖥️ System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.10 – 3.11 |
| RAM | 4 GB | 8 GB+ |
| GPU | Not required | CUDA GPU for faster inference |
| OS | Windows / macOS / Linux | Any |

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8 model and inference |
| `streamlit` | Interactive web application UI |
| `opencv-python` | Video and image processing |
| `Pillow` | Image loading and manipulation |
| `torch` | Deep learning backend |

---

## 📄 Reports & Documentation

All project reports are in the `docs/` folder:

- [`ASSIGNMENT_1_REPORT.md`](docs/ASSIGNMENT_1_REPORT.md) — Data pipeline, baseline model, evaluation
- [`Assignment_2_Report.md`](docs/Assignment_2_Report.md) — Extended analysis
- [`FINAL_REPORT.md`](docs/FINAL_REPORT.md) — Final project summary
- [`colab_training_guide.md`](docs/colab_training_guide.md) — How to train on Google Colab (free GPU)

---

## 🤝 Contributing

1. Fork this repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📜 License

This project is for academic purposes — **BS Computer Science, Computer Vision Assignment**.

---

## 👤 Author

**Shahzeb**  
BS-CS-07 | Computer Vision Course  
GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
