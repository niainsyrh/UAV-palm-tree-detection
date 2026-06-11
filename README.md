# UAV Palm Tree Health Detection 
A web application for detecting and classifying palm tree health conditions from UAV (drone) images using YOLOv8, integrated with Azure cloud services and Power BI for enterprise-level plantation monitoring.

## Overview
This app analyzes UAV images to detect and classify palm trees into 4 health categories:
- **Dead** — Dead palm trees
- **Healthy** — Healthy palm trees  
- **Small** — Small/young palm trees
- **Yellow** — Yellowing palm trees

## Features
- Single image upload for instant analysis
- **Batch processing** — upload multiple images at once
- **ZIP file support** — upload entire folder as ZIP
- Annotated result image with bounding boxes
- Detection count and confidence scores per class
- PDF report generation with results summary
- Excel export for batch results
- **Azure Blob Storage** — automatic image archiving to cloud
- **Azure SQL Database** — persistent storage of all batch results
- **Power BI Dashboard** — plantation health trend monitoring

## Tech Stack
- **Backend:** Python, Flask
- **Detection Model:** YOLOv8 (Ultralytics) — mAP50: 0.995
- **Image Processing:** OpenCV, Pillow
- **Report Generation:** FPDF, OpenPyXL
- **Frontend:** HTML, CSS, JavaScript
- **Cloud Storage:** Azure Blob Storage
- **Database:** Azure SQL Database
- **BI Dashboard:** Microsoft Power BI

## System Architecture
Field Worker (Website)          Manager (Power BI)
↓                               ↑
Flask Web App                   Azure SQL Database
↓                               ↑
YOLO Model (inference)    →    Batch Results Saved
↓
Azure Blob Storage (images archived)

## API Routes
| Route | Method | Description |
|---|---|---|
| `/predict` | POST | Single image detection |
| `/predict_batch` | POST | Multiple images batch |
| `/batch` | POST | ZIP file batch upload |
| `/history` | GET | All batch results from Azure SQL |
| `/report/<uid>` | GET | Download PDF report |
| `/report_batch_excel` | POST | Download Excel export |

## Project Structure
tree_health_app/
├── app.py              # Main Flask application
├── .env                # Environment variables (not in repo)
├── .env.example        # Environment variables template
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Frontend UI
├── uploads/            # Uploaded images (auto-created)
├── results/            # Detection results (auto-created)
└── README.md

## Setup & Installation
### Prerequisites
- Python 3.8+
- Anaconda (recommended)
- YOLOv8 trained model weights (`best.pt`)
- Azure account (Blob Storage + SQL Database)

### Install dependencies
```bash
pip install flask ultralytics opencv-python pillow fpdf2 numpy azure-storage-blob pymssql python-dotenv openpyxl
```

### Environment variables
Create a `.env` file in the project root (see `.env.example`):
AZURE_CONNECTION_STRING=your_blob_connection_string
SQL_SERVER=your_server.database.windows.net
SQL_DATABASE=your_database_name
SQL_USER=your_username
SQL_PASSWORD=your_password

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
### Single Image
1. Select **Single image** mode
2. Upload a UAV image
3. Click **Analyse**
4. View results and download PDF

### Batch Processing
1. Select **Batch upload** mode
2. Upload multiple images
3. Click **Analyse**
4. Results automatically saved to Azure SQL
5. Images archived to Azure Blob Storage
6. View trends in Power BI dashboard

## Training
Training notebooks are available in the `training/` folder (`.ipynb` files).
