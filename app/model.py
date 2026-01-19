import os
import requests
import torch
import torch.nn as nn
from torchvision import models

# =========================
# CONFIG
# =========================
MODEL_URL = os.getenv("MODEL_URL")   # URL .pth จาก Supabase
MODEL_PATH = "model.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_MODEL = None


# =========================
# Download model from Supabase
# =========================
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

    print("✅ Model downloaded successfully")


# =========================
# Build model architecture
# =========================
def build_model():
    """
    ⚠️ IMPORTANT
    Architecture ต้องตรงกับตอน train 100%
    โมเดลนี้ train มาเป็น 3-class
    """
    model = models.mobilenet_v3_large(weights=None)

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        3   # ✅ ตรงกับ checkpoint
    )

    return model


# =========================
# Load model
# =========================
def load_model():
    print("🚀 Preparing model...")

    download_model()

    print("🧠 Building model architecture")
    model = build_model()

    print("📂 Loading model weights (.pth)")
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    print(f"✅ Model loaded successfully on {DEVICE}")
    return model


# =========================
# Singleton
# =========================
def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL
