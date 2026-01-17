import os
import requests
import torch

MODEL_URL = os.getenv("MODEL_URL")   # public URL จาก Supabase
MODEL_PATH = "model.pt"
DEVICE = "cpu"

_MODEL = None


def download_model():
    if os.path.exists(MODEL_PATH):
        return

    if not MODEL_URL:
        raise RuntimeError("MODEL_URL is not set")

    print("⬇️ Downloading TorchScript model from Supabase...")
    r = requests.get(MODEL_URL, stream=True, timeout=60)
    r.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)

    print("✅ Model downloaded")


def load_model():
    download_model()

    # ✅ TorchScript ต้องใช้ jit.load เท่านั้น
    model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
    model.eval()
    return model


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL
