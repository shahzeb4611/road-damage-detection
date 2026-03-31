# Google Colab Setup for YOLOv8 Training

To train your road damage detection model on Google Colab, you'll need the following steps:

1. **Upload the zipped dataset:** Upload your zipped images folder to your Google Drive. Let's assume you name it `dataset.zip`.
2. **Create a new notebook in Google Colab:** Go to Google Colab and create a new Python 3 notebook.
3. **Change the Runtime to GPU:** Go to `Runtime` -> `Change runtime type` -> select `T4 GPU` (or better if available) -> `Save`.

## Steps in Colab Notebook

You will run the following cells in your Colab notebook one by one:

### Step 1: Mount Google Drive

This allows Colab to access your files on Drive.

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Unzip the Dataset

Extract your dataset to the Colab environment. Change `MyDrive/dataset.zip` to the actual path of your zip file on your Drive.

```bash
!unzip -q /content/drive/MyDrive/dataset.zip -d /content/dataset
```

### Step 3: Install Ultralytics YOLOv8

Install the necessary library.

```bash
!pip install ultralytics
```

### Step 4: Create the data.yaml file

YOLO needs a `yaml` file to tell it where the images and labels are, and what the classes are.

* **Important:** You need to adjust the `train` and `val` paths below based on the structure inside your unzipped folder. Assuming your zip contains `images/train` and `images/val`.

```python
import yaml

data_config = {
    'path': '/content/dataset', # Path to the unzipped root directory
    'train': 'images/train',    # Relative path from 'path' to training images
    'val': 'images/val',        # Relative path from 'path' to validation images
    'names': {
        0: 'longitudinal_crack',
        1: 'transverse_crack',
        2: 'alligator_crack',
        3: 'pothole'
    }
}

with open('/content/dataset/data.yaml', 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False)

print("data.yaml created successfully.")
```

### Step 5: Start Training

This is the command to train your model. We are using `yolov8s.pt` (Small model) or you can try `yolov8m.pt` (Medium) for better accuracy, longer epochs (`50` or `100`), and a larger image size (`640`).

```bash
# Using Python API is generally cleaner in Notebooks
from ultralytics import YOLO

# Load a pre-trained model (we upgrade from nano 'n' to small 's' for better accuracy)
model = YOLO('yolov8s.pt')

# Train the model
results = model.train(
    data='/content/dataset/data.yaml',
    epochs=100,      # Increase epochs for better accuracy (try 50 or 100)
    imgsz=640,       # Larger image size helps detect small cracks (default is 640 anyway)
    batch=16,        # Adjust batch size based on GPU memory (usually 16 or 32 is fine)
    name='rdd_colab_run',
    device=0         # 0 means use the first GPU
)
```

### Step 6: Download the Best Model

Once training is done, your best model weights (`best.pt`) will be saved in the `runs/detect/rdd_colab_run/weights/` directory.

You can download it using this code snippet, or just find it in the file explorer on the left sidebar in Colab and download it manually.

```python
from google.colab import files
files.download('/content/runs/detect/rdd_colab_run/weights/best.pt')
```

## How to get >80% accuracy?

Achieving over 80% mAP (mean Average Precision) can be challenging depending on the complexity of the dataset.

* **Model Size:** The `yolov8s` model is a good step up from `yolov8n`. If `yolov8s` still doesn't reach your target, try `yolov8m`.
* **Data Quality:** Make sure your dataset has accurate bounding boxes.
* **Image Size:** `imgsz=640` or even `imgsz=1280` will help the model detect tiny, thin cracks better, but uses more GPU memory.
* **Epochs:** Train it longer (`epochs=100` or `150`). Let `patience=50` (early stopping) stop the training if it's not improving.
