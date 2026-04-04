# 🛣️ Road Damage Detection System (RDDSS)

An AI-powered computer vision application for real-time detection of road damages (Potholes, Cracks, etc.) using **YOLOv8s** and **Streamlit**.

## 🚦 Project Overview
This project aims to automate road maintenance surveys by detecting and classifying road surface damages. The system is trained on a balanced subset of the RDD2022 dataset and provides a high-performance, user-friendly interface for image and video analysis.

---

## 📈 Model Performance (YOLOv8s)
The model was trained for **50 epochs** with an image size of **640x640**. Below are the key performance metrics achieved on the validation set:

| Metric | Accuracy / Value |
| :--- | :--- |
| **Precision** | 49.1% |
| **Recall** | 37.6% |
| **mAP@50** | 36.4% |
| **mAP@50-95** | 15.5% |

> [!NOTE]
> The performance metrics show significant improvement after switching to the `yolov8s.pt` architecture, providing a balanced trade-off between speed and accuracy.

---

## ✨ Key Features
-   🖼️ **Multi-Image Scanning**: Detect damages in multiple road images at once.
-   🎬 **Video Processing**: Full video annotation with frame-by-frame seeking and playback speed control.
-   📷 **Live Webcam**: Real-time road damage detection feed for mobile or vehicle-mounted cameras.
-   💎 **Premium UI**: Dark-themed glassmorphism interface built with Streamlit.
-   📥 **Downloadable Reports**: Export annotated images and videos directly from the dashboard.

---

## 🛠️ Installation & Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/shahzeb4611/Road-Damage-Detection.git
    cd Road-Damage-Detection
    ```

2.  **Create Virtual Environment**:
    ```bash
    python -m venv venv
    venv\Scripts\activate  # On Windows
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application**:
    ```bash
    streamlit run app.py
    ```

---

## 📂 Project Structure
```text
Road-Damage-Detection/
├── app.py                # Main Streamlit dashboard
├── main.py               # Alternative entry point
├── scripts/              # Training & maintenance scripts
│   ├── train.py          # Script for intense model training
│   └── resume_train.py   # Utility to resume training
├── models/               # Stored YOLO model weights
│   └── road_damage_best.pt
├── data/                 # Configuration yaml files
├── .gitignore            # Exclude large data/logs
└── README.md             # Project documentation
```

---

## ⚖️ Credits
-   **Dataset**: RDD2022 (Road Damage Detection)
-   **Model Architecture**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
-   **UI Framework**: [Streamlit](https://streamlit.io/)
