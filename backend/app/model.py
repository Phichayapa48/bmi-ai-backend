import os
import requests
import torch
import torch.nn as nn
from torchvision import models

# =========================
# Config
# =========================
MODEL_URL = os.getenv("MODEL_URL")      # URL .pth จาก Supabase
MODEL_PATH = "model.pth"
DEVICE = torch.device("cpu")

_MODEL = None


# =========================
# Download model
# =========================
def download_model():
    if os.path.exists(MODEL_PATH):
        print("📦 Model already exists")
        return

    if not MODEL_URL:
        raise RuntimeError("❌ MODEL_URL is not set")

    print("⬇️ Downloading model (.pth)...")
    r = requests.get(MODEL_URL, stream=True, timeout=60)
    r.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print("✅ Model downloaded successfully")


# =========================
# Build model architecture
# =========================
def build_model():
    model = models.mobilenet_v3_large(weights=None)

    # Regression head (BMI)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, 1
    )

    return model


# =========================
# Load model
# =========================
def load_model():
    download_model()

    print("🧠 Loading PyTorch model (.pth)...")

    model = build_model()

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # 🔧 รองรับกรณีมี prefix เช่น "module."
    if any(k.startswith("module.") for k in state_dict.keys()):
        print("⚠️ Removing 'module.' prefix from state_dict")
        state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE)
    model.eval()

    print("✅ Model loaded and ready")
    return model


# =========================
# Singleton accessor
# =========================
def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL
