"""Run: streamlit run app.py"""
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps
from ultralytics import YOLO

st.set_page_config(page_title='Open-Vocabulary Object Detection', layout='wide')
st.title('Open-Vocabulary Object Detection')
st.caption('Standard YOLO vs. YOLO-World | Compare the same scene with custom text prompts')

@st.cache_resource(max_entries=4)
def load_model(weights, device):
    # Shared cached models are mutable: protect set_classes AND prediction together.
    model = YOLO(weights)
    model.to(device)
    return model, RLock()

def parse_prompts(text):
    return list(dict.fromkeys(p.strip() for p in re.split(r'[,\n;]+', text) if p.strip()))

def predict(weights, device, image, conf, iou, size, prompts=None):
    model, lock = load_model(weights, device)
    with lock:
        if prompts is not None:
            model.set_classes(prompts)
        result = model.predict(source=image, conf=conf, iou=iou, imgsz=size,
                               device=device, verbose=False, agnostic_nms=True)[0]
        rows = []
        if result.boxes is not None:
            for k, box in enumerate(result.boxes, 1):
                label = result.names[int(box.cls.item())]
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                rows.append({'Detection': k, 'Class / matched prompt': label,
                             'Confidence': round(float(box.conf.item()), 4),
                             'x1': round(x1, 1), 'y1': round(y1, 1),
                             'x2': round(x2, 1), 'y2': round(y2, 1)})
        # Ultralytics plot() returns BGR; Streamlit receives RGB.
        annotated = result.plot(line_width=2)[:, :, ::-1].copy()
    return annotated, rows

def png_bytes(array):
    output = io.BytesIO()
    Image.fromarray(array).save(output, format='PNG')
    return output.getvalue()

world_weights = st.sidebar.selectbox('YOLO-World model',
    ['yolov8s-worldv2.pt', 'yolov8s-world.pt', 'yolov8m-worldv2.pt'])
standard_conf = st.sidebar.slider('Standard YOLO confidence', 0.05, 0.80, 0.25, 0.01)
world_conf = st.sidebar.slider('YOLO-World confidence', 0.05, 0.80, 0.15, 0.01)
iou = st.sidebar.slider('NMS IoU (both models)', 0.30, 0.80, 0.50, 0.05)
size = st.sidebar.selectbox('Inference image size', [640, 960, 1280])
devices = ['cpu']
if torch.cuda.is_available():
    devices.insert(0, 'cuda:0')
if torch.backends.mps.is_available():
    devices.append('mps')
device = st.sidebar.selectbox('Device', devices)
st.sidebar.caption('CPU is the safe starting option on Mac. First run downloads model and text-encoder weights; internet is required.')

prompt_text = st.text_area('Custom class prompts (comma-separated)',
    'eyeglasses, ballpoint pen, headphones, wristwatch, USB charger, whiteboard marker, ID badge',
    help='Use labels for objects actually in the photo. Change one phrase at a time for a controlled comparison.')
prompts = parse_prompts(prompt_text)
mode = st.radio('Image source', ['Upload image', 'Camera capture'], horizontal=True)
source = (st.file_uploader('Choose a desk or everyday scene', type=['jpg', 'jpeg', 'png', 'webp'])
          if mode == 'Upload image' else st.camera_input('Take a photo'))

if source is None:
    st.info('Upload or capture one real scene containing at least five distinct non-COCO object types.')
    st.stop()
try:
    raw = source.getvalue()
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert('RGB')
    image.thumbnail((2400, 2400))
except Exception as exc:
    st.error(f'Could not read this image: {exc}')
    st.stop()
with st.expander('Original scene'):
    st.image(image, width=650)

settings = {'image_sha256': hashlib.sha256(raw).hexdigest(), 'world_model': world_weights,
            'standard_model': 'yolov8n.pt', 'prompts': prompts, 'standard_conf': standard_conf,
            'world_conf': world_conf, 'iou': iou, 'imgsz': size, 'device': device,
            'image_width': image.width, 'image_height': image.height}
signature = json.dumps(settings, sort_keys=True)
if st.button('Run comparison', type='primary'):
    st.session_state.pop('result', None)
    if not prompts or len(prompts) > 50:
        st.error('Enter between 1 and 50 non-empty class prompts.')
    else:
        try:
            with st.spinner('Loading weights and detecting objects in both models...'):
                standard_img, standard_rows = predict('yolov8n.pt', device, image, standard_conf, iou, size)
                world_img, world_rows = predict(world_weights, device, image, world_conf, iou, size, prompts)
            record = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), **settings,
                      'standard_detections': standard_rows, 'world_detections': world_rows}
            st.session_state['result'] = {'signature': signature, 'record': record,
                                         'standard_img': standard_img, 'world_img': world_img}
            st.session_state.setdefault('history', []).append(record)
        except Exception as exc:
            st.error(f'Detection failed: {exc}')
            st.info('Check internet for weight downloads and install requirements.txt. If MPS fails, select CPU and run again.')

saved = st.session_state.get('result')
if saved and saved['signature'] != signature:
    st.info('Settings or image changed. Click Run comparison to refresh the results.')
elif saved:
    record = saved['record']
    left, right = st.columns(2)
    for col, key, title, weights, threshold in [
        (left, 'standard', 'Standard YOLO (COCO)', 'yolov8n.pt', standard_conf),
        (right, 'world', 'YOLO-World (custom prompts)', world_weights, world_conf)]:
        with col:
            st.subheader(title)
            st.caption(f'{weights} | conf={threshold:.2f} | IoU={iou:.2f} | imgsz={size}')
            st.image(saved[key + '_img'], use_container_width=True)
            rows = record[key + '_detections']
            st.write(f'{len(rows)} boxes; {len(set(r["Class / matched prompt"] for r in rows))} distinct predicted labels')
    st.subheader('Custom prompts and detection results')
    st.code('model.set_classes(' + json.dumps(prompts) + ')', language='python', wrap_lines=True)
    frames = []
    for key, name in [('standard', 'Standard YOLO'), ('world', 'YOLO-World')]:
        for row in record[key + '_detections']:
            frames.append({'Model': name, **row})
        if not record[key + '_detections']:
            st.info(f'{name}: no detections at the current confidence threshold.')
    table = pd.DataFrame(frames, columns=['Model', 'Detection', 'Class / matched prompt', 'Confidence', 'x1', 'y1', 'x2', 'y2'])
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption('Coordinates refer to the displayed, orientation-corrected input image. A predicted label is not proof of a correct detection: inspect every box. Five labels or five boxes do not automatically mean five distinct non-COCO objects.')
    with st.expander('Download results'):
        st.download_button('Standard annotated image', png_bytes(saved['standard_img']), 'standard_yolo.png', 'image/png')
        st.download_button('YOLO-World annotated image', png_bytes(saved['world_img']), 'yolo_world.png', 'image/png')
        st.download_button('Detection table CSV', table.to_csv(index=False).encode(), 'detections.csv', 'text/csv')
        st.download_button('Current settings and detections JSON', json.dumps(record, indent=2), 'run_results.json', 'application/json')

history = st.session_state.get('history', [])
if history:
    with st.expander('Experiment history (this browser session)'):
        st.caption('Keep the scene, model, and thresholds fixed when testing prompt wording. Confidence is not measured accuracy and is not calibrated equally across models.')
        for n, run in enumerate(history, 1):
            st.write(f'Run {n}: {run["world_model"]}, conf={run["world_conf"]:.2f}')
            st.write(', '.join(run['prompts']))
            st.dataframe(pd.DataFrame(run['world_detections']), hide_index=True)
        st.download_button('Download all experiments', json.dumps(history, indent=2), 'experiment_history.json', 'application/json')

with st.expander('Python script'):
    script = Path(__file__).read_text(encoding='utf-8')
    st.download_button('Download app.py', script, 'app.py', 'text/x-python')
    st.code(script, language='python')
