from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import traceback
import torch

from app.model import get_model
from app.utils import preprocess_image
from app.face_utils import detect_and_crop_face
from app.quality_check import quality_check

app = FastAPI()

model = None

LABELS = ["under", "normal", "over"]

BMI_MAP = {
    "under": 17.5,
    "normal": 22.0,
    "over": 27.5
}

@app.get("/")
def health():
    return {"status": "ok", "service": "BMI AI Backend"}

@app.on_event("startup")
def startup_event():
    global model
    print("🚀 Loading model...")
    model = get_model()
    model.eval()
    print("✅ Model ready")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        if not image_bytes:
            return {"error": "empty_file"}

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        ok, reason = quality_check(image)
        if not ok:
            print(f"⚠️ Quality warning: {reason}")

        face_image = detect_and_crop_face(image)
        if face_image is None:
            face_image = image

        x = preprocess_image(face_image)

        with torch.no_grad():
            x = x.to(next(model.parameters()).device)

            logits = model(x)

            temperature = 1.5  # 🔥 แก้ปัญหาปัด under
            probs = torch.softmax(logits / temperature, dim=1)

            cls_idx = int(probs.argmax(dim=1).item())
            cls_name = LABELS[cls_idx]
            confidence = float(probs[0, cls_idx])

        return {
            "category": cls_name,
            "confidence": round(confidence, 3),
            "bmi_estimate": BMI_MAP[cls_name],
            "message": "success"
        }

    except Exception:
        traceback.print_exc()
        return {
            "error": "prediction_failed",
            "message": "ไม่สามารถประเมิน BMI จากภาพนี้ได้"
        }
