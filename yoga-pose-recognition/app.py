import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf

APP_DIR = Path(__file__).resolve().parent
IMG_SIZE = 160

st.set_page_config(page_title="Yoga 107 Classifier", page_icon="🧘", layout="wide")
st.title("Yoga 107 Pose Classifier")
st.caption("Upload a yoga image and compare quantized EfficientNetB0 model versions")

@st.cache_resource
def load_interpreter(filename):
    interpreter = tf.lite.Interpreter(model_path=str(APP_DIR / "models" / filename), num_threads=4)
    interpreter.allocate_tensors()
    return interpreter

@st.cache_data
def load_classes():
    return json.loads((APP_DIR / "classes.json").read_text())

def prepare(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
    canvas.paste(image, ((IMG_SIZE-image.width)//2, (IMG_SIZE-image.height)//2))
    return np.asarray(canvas, dtype=np.float32)

def predict(interpreter, image):
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    interpreter.set_tensor(inp["index"], image[None].astype(inp["dtype"]))
    interpreter.invoke()
    return interpreter.get_tensor(out["index"])[0]

model_options = {
    "Fine tuned EfficientNetB0 float16": "efficientnet_fine_tuned_float16.tflite",
    "Frozen backbone EfficientNetB0 float16": "efficientnet_head_float16.tflite",
}
choice = st.sidebar.selectbox("Model version", list(model_options))
st.sidebar.write("Both files use float16 weight quantization and a 160 × 160 RGB input.")
uploaded = st.file_uploader("Upload a test image", type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded:
    original = Image.open(uploaded)
    x = prepare(original)
    interpreter = load_interpreter(model_options[choice])
    scores = predict(interpreter, x)
    classes = load_classes()
    order = np.argsort(scores)[::-1][:5]

    left, right = st.columns([1.05, .95])
    with left:
        st.image(original, caption="Uploaded image", use_container_width=True)
    with right:
        st.subheader("Prediction")
        st.metric("Predicted pose", classes[int(order[0])].replace("_", " ").title())
        st.metric("Confidence", f"{scores[order[0]]:.1%}")
        rows = [{"Rank": i+1, "Yoga pose": classes[int(k)].replace("_", " ").title(), "Confidence": float(scores[k])} for i,k in enumerate(order)]
        st.dataframe(rows, hide_index=True, use_container_width=True, column_config={"Confidence": st.column_config.ProgressColumn(format="%.2f", min_value=0., max_value=1.)})
        st.bar_chart({classes[int(k)]: float(scores[k]) for k in order}, horizontal=True)
else:
    st.info("Choose a clear test image that was not used for training.")

with st.expander("Model details"):
    st.write("The backbone is ImageNet pretrained EfficientNetB0. The first version trains only a new 107 class head. The best version also fine tunes the top 35 layers with a smaller learning rate. Batch normalization layers remain frozen during fine tuning.")
