# UAV Palm Tree Health Detection 

A web application for detecting and classifying palm tree health conditions from UAV (drone) images using YOLOv8.

## Overview

This app analyzes UAV images to detect and classify palm trees into 4 health categories:
- **Dead** — Dead palm trees
- **Healthy** — Healthy palm trees  
- **Small** — Small/young palm trees
- **Yellow** — Yellowing palm trees

## Features

- Upload UAV images for instant analysis
- Annotated result image with bounding boxes
- Detection count and confidence scores per class
- PDF report generation with results summary

## Tech Stack

- **Backend:** Python, Flask
- **Detection Model:** YOLOv8 (Ultralytics)
- **Image Processing:** OpenCV, Pillow
- **Report Generation:** FPDF
- **Frontend:** HTML, CSS, JavaScript

## Project Structure

```
tree_health_app/
├── app.py              # Main Flask application
├── templates/
│   └── index.html      # Frontend UI
├── uploads/            # Uploaded images (auto-created)
├── results/            # Detection results (auto-created)
└── README.md
```

## Setup & Installation

### Prerequisites
- Python 3.8+
- Anaconda (recommended)
- YOLOv8 trained model weights (`best.pt`)

### Install dependencies

```bash
pip install flask ultralytics opencv-python pillow fpdf2 numpy
```

### Update model path

In `app.py`, update the model path to your trained weights:

```python
MODEL_PATH = r'path/to/your/weights/best.pt'
```

### Run the app

```bash
python app.py
```

Open your browser and go to: `http://localhost:5000`

## Usage

1. Open the web app in your browser
2. Upload a UAV image
3. View detection results and annotated image
4. Download PDF report

## Training

Training notebooks are available in the `training/` folder (`.ipynb` files).


