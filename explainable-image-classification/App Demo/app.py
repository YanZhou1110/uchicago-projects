import os
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    from torchvision import models, transforms
except Exception as e:
    models = None
    transforms = None
    TORCHVISION_ERROR = e
else:
    TORCHVISION_ERROR = None

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR.parent / "Data & Models"
RESNET_PATH = MODEL_DIR / "finetuned_resnet18_pneumoniamnist.pth"
BASELINE_PATH = MODEL_DIR / "baseline_cnn_pneumoniamnist.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
LABEL_NAMES = {0: "Normal", 1: "Pneumonia"}


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_resnet18():
    if models is None:
        raise RuntimeError(f"torchvision could not be imported: {TORCHVISION_ERROR}")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def load_checkpoint(model_name):
    if model_name == "Fine-Tuned ResNet18":
        path = RESNET_PATH
        model = build_resnet18()
        target_layer = model.layer4[-1]
    else:
        path = BASELINE_PATH
        model = BaselineCNN(num_classes=2)
        target_layer = model.features[12]

    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")

    checkpoint = torch.load(path, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, target_layer


if transforms is not None:
    # Match the notebook training/evaluation preprocessing.
    # The ResNet18 model was fine-tuned using ImageNet normalization.
    preprocess = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
else:
    preprocess = None


def prepare_image(image):
    if image is None:
        raise ValueError("Please upload a chest X-ray image first.")
    if preprocess is None:
        raise RuntimeError(f"torchvision transforms could not be imported: {TORCHVISION_ERROR}")
    image = Image.fromarray(image).convert("L")
    rgb_image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    display_img = np.asarray(rgb_image).astype(np.float32) / 255.0
    return input_tensor, display_img


def triage_recommendation(pred_class, confidence):
    if pred_class == 1 and confidence >= 90:
        return "High Priority", "Prioritize for radiologist review."
    if pred_class == 1 and confidence >= 70:
        return "Routine Review (Possible Pneumonia)", "Standard queue, but prioritize above clearly normal cases."
    if pred_class == 0 and confidence < 70:
        return "Uncertain / Manual Review", "Manual verification required because model confidence is low."
    return "Low Priority", "No urgent model-flagged action. Keep normal clinical review process."


def compute_gradcam(model, target_layer, input_tensor, target_class):
    activations = []
    gradients = []

    def forward_hook(module, inputs, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    f_handle = target_layer.register_forward_hook(forward_hook)
    b_handle = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad(set_to_none=True)
    output = model(input_tensor)
    score = output[:, target_class].sum()
    score.backward()

    f_handle.remove()
    b_handle.remove()

    activation = activations[0]
    gradient = gradients[0]
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activation).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def colorize_cam(cam):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("jet")
    heatmap = cmap(cam)[..., :3]
    return heatmap


def predict_and_explain(image, model_name):
    try:
        model, target_layer = load_checkpoint(model_name)
        input_tensor, display_img = prepare_image(image)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)[0]
            confidence, pred = torch.max(probs, dim=0)

        pred_class = int(pred.item())
        confidence_pct = float(confidence.item() * 100)
        risk_level, recommendation = triage_recommendation(pred_class, confidence_pct)

        cam = compute_gradcam(model, target_layer, input_tensor, pred_class)
        heatmap = colorize_cam(cam)
        overlay = np.clip(0.55 * display_img + 0.45 * heatmap, 0, 1)

        summary = (
            f"### Prediction Summary\n"
            f"**Model:** {model_name}\n\n"
            f"**Prediction:** {LABEL_NAMES[pred_class]}\n\n"
            f"**Confidence:** {confidence_pct:.2f}%\n\n"
            f"**Risk Level:** {risk_level}\n\n"
            f"**Recommendation:** {recommendation}\n\n"
            f"This tool is a triage demo only. It supports clinical review; it does not replace a doctor or radiologist."
        )
        return summary, (heatmap * 255).astype(np.uint8), (overlay * 255).astype(np.uint8)
    except Exception as e:
        return f"### Error\n{type(e).__name__}: {e}", None, None


with gr.Blocks(title="Explainable PneumoniaMNIST Triage Demo") as demo:
    gr.Markdown("# Explainable Medical Image Triage Demo")
    gr.Markdown("Upload a chest X-ray style image. The app returns a model prediction, confidence score, triage recommendation, and Grad-CAM explanation.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload Chest X-ray", type="numpy")
            model_choice = gr.Radio(
                choices=["Fine-Tuned ResNet18", "Baseline CNN"],
                value="Fine-Tuned ResNet18",
                label="Model"
            )
            run_button = gr.Button("Analyze Image")
        with gr.Column(scale=1):
            output_text = gr.Markdown(label="Prediction")
            heatmap_output = gr.Image(label="Grad-CAM Heatmap")
            overlay_output = gr.Image(label="Grad-CAM Overlay")

    run_button.click(
        fn=predict_and_explain,
        inputs=[image_input, model_choice],
        outputs=[output_text, heatmap_output, overlay_output]
    )

if __name__ == "__main__":
    demo.launch()
