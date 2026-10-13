"""Optional web demo:  python app.py   (needs `pip install gradio`)"""
import gradio as gr

import config
from dataset import build_transforms
from model import load_checkpoint
from predict import predict_image
from utils import get_device

device = get_device()
model, ck = load_checkpoint(config.CKPT_DIR / "best.pt", device)
_, tf = build_transforms(ck["img_size"])


def classify(img):
    label, conf, probs = predict_image(model, img, tf, device)
    return probs


gr.Interface(fn=classify, inputs=gr.Image(type="pil"), outputs=gr.Label(num_top_classes=2),
             title="Deepfake / Real Image Detector",
             description="Upload an image to estimate whether it is REAL or FAKE.").launch()
