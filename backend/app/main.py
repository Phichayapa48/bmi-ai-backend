from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import traceback

from app.model import get_model
from app.utils import preprocess_image

app = FastAPI()

# =========================
# Health Check (Render)
# =========================
@app.get("/")
def health():
    return {"status": "ok", "service": "BMI AI Backend"}

# =========================
# Load model on startup
# =========================
@app.on_event("startup")
def startup_event():
    print("🚀 Loading model...")
    get_model()
    print("✅ Model ready")

# =========================
# Predict BMI
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1️⃣ อ่านไฟล์ภาพ
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 2️⃣ preprocess
        x = preprocess_image(image)

        # 3️⃣ inference
        model = get_model()
        with torch.no_grad():
            y = model(x)
            bmi = float(y.squeeze().item())

        return {
            "bmi": bmi
        }

    except Exception as e:
        print("❌ Predict error")
        traceback.print_exc()
        return {
            "error": "prediction_failed",
            "message": str(e)
        }
