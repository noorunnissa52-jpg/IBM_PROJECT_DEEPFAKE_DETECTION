from pathlib import Path

import io
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

import config
from dataset import build_transforms
from model import load_checkpoint
from predict import predict_image
from utils import get_device

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"

app = FastAPI(title="Deepfake Detector API")

# Load the trained model once at startup.
device = get_device()
model, ckpt = load_checkpoint(str(config.CKPT_DIR / "best.pt"), device)
_, tf = build_transforms(ckpt["img_size"])


@app.get("/")
def read_root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        label, conf, probs = predict_image(model, image, tf, device)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

    return {
        "labelName": label,
        "confidenceValue": conf,
        "probabilities": {
            key: float(value)
            for key, value in probs.items()
        },
    }


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
