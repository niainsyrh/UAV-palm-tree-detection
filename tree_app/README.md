# MOPAD Tree Health Web App

A professional Flask web app for detecting tree health from UAV imagery using YOLOv8.

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Make sure your model path in `app.py` is correct:
```python
MODEL_PATH = r'C:\Users\nia.insyirah\models\yolo_mopad\weights\best.pt'
```

3. Run the app:
```
python app.py
```

4. Open your browser at: http://localhost:5000

## Features
- Upload UAV images via drag & drop
- AI detection of Dead, Healthy, Small, Yellow trees
- Live stat cards with confidence scores
- Annotated image with bounding boxes
- Downloadable PDF report
