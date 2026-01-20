import os
import requests
import torch
import torch.nn as nn
from torchvision import models

MODEL_URL = os.getenv("MODEL_URL")
MODEL_PATH = "model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_MODEL = None

def download_model():
    if os.path.exists(MODEL_PATH):
        print("📦 Model file already exists")
        return

    if not MODEL_URL:
        raise RuntimeError("❌ MODEL_URL is not set")

    print(f"⬇️ Downloading model from: {MODEL_URL}")
    r = requests.get(MODEL_URL, stream=True, timeout=60)
    r.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)

    print("✅ Model downloaded")

def build_model():
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, 3
    )
    return model

def load_model():
    download_model()
    model = build_model()

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224).to(DEVICE)
        assert model(dummy).shape[-1] == 3

    print(f"✅ Model loaded on {DEVICE}")
    return model

def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL
