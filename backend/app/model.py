# app/model.py
import os
import requests
import torch

MODEL_URL  = os.getenv("MODEL_URL")   # Supabase public URL
MODEL_PATH = "model.pt"
DEVICE = "cpu"

_MODEL = None


def download_model():
    if os.path.exists(MODEL_PATH):
        print("📦 Model already exists")
        return

    if not MODEL_URL:
        raise RuntimeError("❌ MODEL_URL is not set")

    print(f"⬇️ Downloading model from: {MODEL_URL}")

    with requests.get(MODEL_URL, stream=True, timeout=120) as r:
        r.raise_for_status()

        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    # 🔍 sanity check
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"✅ Model downloaded ({size_mb:.2f} MB)")

    if size_mb < 1:
        raise RuntimeError("❌ Model file looks corrupted (too small)")


def load_model():
    download_model()

    print("🧠 Loading TorchScript model...")
    model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
    model.eval()
    print("✅ Model loaded")
    return model


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL
