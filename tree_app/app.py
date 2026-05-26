from flask import Flask, render_template, request, jsonify, send_file
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
import uuid
import json
from datetime import datetime
from fpdf import FPDF
import base64
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# ── paths ──────────────────────────────────────────────────────────────
MODEL_PATH  = r'C:\Users\nia.insyirah\models\yolo_mopad\weights\best.pt' #
UPLOAD_DIR  = os.path.join(os.path.dirname(__file__), 'uploads')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(UPLOAD_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASS_COLORS = {
    'Dead':    '#ef4444',
    'Healthy': '#22c55e',
    'Small':   '#f59e0b',
    'Yellow':  '#eab308',
}

model = YOLO(MODEL_PATH)

# ── helpers ─────────────────────────────────────────────────────────────
def img_to_b64(arr_bgr):
    _, buf = cv2.imencode('.jpg', arr_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf).decode()


def run_inference(img_path):
    results  = model.predict(source=img_path, imgsz=640, conf=0.25, verbose=False)
    result   = results[0]
    annotated = result.plot()                      # BGR numpy array

    counts = {'Dead': 0, 'Healthy': 0, 'Small': 0, 'Yellow': 0}
    confs  = {'Dead': [], 'Healthy': [], 'Small': [], 'Yellow': []}

    for box in result.boxes:
        cls_id   = int(box.cls[0])
        cls_name = result.names[cls_id]
        conf     = float(box.conf[0])
        if cls_name in counts:
            counts[cls_name] += 1
            confs[cls_name].append(conf)

    avg_conf = {k: (sum(v)/len(v) if v else 0) for k, v in confs.items()}
    total    = sum(counts.values())

    return annotated, counts, avg_conf, total


# ── routes ───────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    ext      = os.path.splitext(file.filename)[1].lower()
    uid      = uuid.uuid4().hex
    src_path = os.path.join(UPLOAD_DIR, f'{uid}{ext}')
    file.save(src_path)

    try:
        annotated, counts, avg_conf, total = run_inference(src_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # save annotated result
    res_path = os.path.join(RESULTS_DIR, f'{uid}_result.jpg')
    cv2.imwrite(res_path, annotated)

    # store metadata for report
    meta = {
        'uid':       uid,
        'filename':  file.filename,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'counts':    counts,
        'avg_conf':  avg_conf,
        'total':     total,
    }
    with open(os.path.join(RESULTS_DIR, f'{uid}_meta.json'), 'w') as f:
        json.dump(meta, f)

    return jsonify({
        'uid':       uid,
        'image_b64': img_to_b64(annotated),
        'counts':    counts,
        'avg_conf':  {k: round(v*100, 1) for k, v in avg_conf.items()},
        'total':     total,
        'timestamp': meta['timestamp'],
        'filename':  file.filename,
    })


@app.route('/report/<uid>')
def download_report(uid):
    meta_path = os.path.join(RESULTS_DIR, f'{uid}_meta.json')
    img_path  = os.path.join(RESULTS_DIR, f'{uid}_result.jpg')
    if not os.path.exists(meta_path):
        return 'Report not found', 404

    with open(meta_path) as f:
        meta = json.load(f)

    pdf = FPDF()
    pdf.add_page()

    # header
    pdf.set_fill_color(22, 101, 52)
    pdf.rect(0, 0, 210, 30, 'F')
    pdf.set_font('Helvetica', 'B', 20)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 12, '', ln=True)
    pdf.cell(0, 12, 'MOPAD Tree Health Report', align='C', ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Helvetica', '', 10)
    pdf.ln(8)
    pdf.cell(0, 6, f"File: {meta['filename']}", ln=True)
    pdf.cell(0, 6, f"Analysis Date: {meta['timestamp']}", ln=True)
    pdf.cell(0, 6, f"Total Trees Detected: {meta['total']}", ln=True)
    pdf.ln(4)

    # table
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_fill_color(220, 252, 231)
    pdf.cell(60, 8, 'Class',      border=1, fill=True)
    pdf.cell(40, 8, 'Count',      border=1, fill=True)
    pdf.cell(40, 8, 'Percentage', border=1, fill=True)
    pdf.cell(40, 8, 'Avg Conf %', border=1, fill=True, ln=True)

    pdf.set_font('Helvetica', '', 10)
    total = meta['total'] or 1
    for cls in ['Dead', 'Healthy', 'Small', 'Yellow']:
        cnt  = meta['counts'].get(cls, 0)
        pct  = round(cnt / total * 100, 1)
        conf = round(meta['avg_conf'].get(cls, 0) * 100, 1)
        pdf.cell(60, 7, cls,       border=1)
        pdf.cell(40, 7, str(cnt),  border=1)
        pdf.cell(40, 7, f'{pct}%', border=1)
        pdf.cell(40, 7, f'{conf}%',border=1, ln=True)

    # annotated image
    if os.path.exists(img_path):
        pdf.ln(6)
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 8, 'Annotated Image:', ln=True)
        pdf.image(img_path, x=10, w=190)

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return send_file(buf, mimetype='application/pdf',
                     as_attachment=True,
                     download_name=f'tree_report_{uid[:8]}.pdf')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
