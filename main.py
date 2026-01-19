from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import traceback
import torch

from app.model import get_model
from app.utils import preprocess_image
from app.face_utils import detect_and_crop_face   # 🔥 เพิ่ม

app = FastAPI()


@app.get("/")
def health():
    return {"status": "ok", "service": "BMI AI Backend"}


@app.on_event("startup")
def startup_event():
    print("🚀 Loading model...")
    get_model()
    print("✅ Model ready")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # =========================
        # 1️⃣ อ่านรูป
        # =========================
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # =========================
        # 2️⃣ ตรวจจับ + crop ใบหน้า
        # =========================
        face = detect_and_crop_face(image)

        # =========================
        # 3️⃣ preprocess ให้ตรงตอน train
        # =========================
        x = preprocess_image(face)

        # =========================
        # 4️⃣ predict
        # =========================
        model = get_model()
        model.eval()

        with torch.no_grad():
            y = model(x)
            bmi = float(y.squeeze().item())

        return {
            "bmi": bmi,
            "message": "success"
        }

    except ValueError as ve:
        # error จาก face detection (ไม่เจอหน้า / หลายหน้า)
        return {
            "error": "invalid_image",
            "message": str(ve)
        }

    except Exception as e:
        print("❌ Predict error")
        traceback.print_exc()
        return {
            "error": "prediction_failed",
            "message": "ไม่สามารถประเมิน BMI จากภาพนี้ได้"
        }
