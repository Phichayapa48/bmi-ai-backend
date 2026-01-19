import os
import requests
import torch
import torch.nn as nn
from torchvision import models

MODEL_URL = os.getenv("MODEL_URL")   # URL .pth จาก Supabase
MODEL_PATH = "model.pth"
DEVICE = "cpu"

_MODEL = None

# =========================
# Download model
# =========================
def download_model():
    if os.path.exists(MODEL_PATH):
        print("📦 Model exists")
        return

    if not MODEL_URL:
        raise RuntimeError("MODEL_URL is not set")

    print("⬇️ Downloading model from:", MODEL_URL)
    r = requests.get(MODEL_URL, stream=True, timeout=60)
    r.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)

    print("✅ Model downloaded")

# =========================
# Build architecture
# =========================
def build_model():
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, 1
    )
    return model

# =========================
# Load model
# =========================
def load_model():
    download_model()

    print("🧠 Loading model (.pth)")
    model = build_model()

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(DEVICE)

    return model

# =========================
# Singleton
# =========================
def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL
