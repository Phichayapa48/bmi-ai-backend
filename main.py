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

# =========================
# GLOBAL MODEL
# =========================
model = None

# =========================
# CLASS CONFIG (ต้องตรงตอน train)
# =========================
BMI_LABELS = {
    0: ("under", 17.5),
    1: ("normal", 22.0),
    2: ("over", 27.5)
}

# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def health():
    return {"status": "ok", "service": "BMI AI Backend"}

# =========================
# LOAD MODEL
# =========================
@app.on_event("startup")
def startup_event():
    global model
    print("🚀 Loading model...")
    model = get_model()
    model.eval()
    print("✅ Model ready")

# =========================
# PREDICT
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1️⃣ Read image
        image_bytes = await file.read()
        if not image_bytes:
            return {
                "error": "empty_file",
                "message": "ไม่พบข้อมูลรูปภาพ"
            }

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 2️⃣ Quality check (soft)
        ok, reason = quality_check(image)
        if not ok:
            print(f"⚠️ Quality warning: {reason}")

        # 3️⃣ Detect face (❗ ต้องเจอเท่านั้น)
        face_image = detect_and_crop_face(image)
        if face_image is None:
            return {
                "error": "no_face_detected",
                "message": "ไม่พบใบหน้าในภาพ กรุณาอัปโหลดภาพใบหน้าคนเท่านั้น"
            }

        # 4️⃣ Preprocess (resize 224)
        x = preprocess_image(face_image)
        x = x.to(next(model.parameters()).device)

        # 5️⃣ Predict (NO temperature / NO trick)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            cls_idx = int(probs.argmax(dim=1).item())
            cls_name, bmi_estimate = BMI_LABELS[cls_idx]
            confidence = float(probs[0, cls_idx])

        # 6️⃣ Response
        return {
            "category": cls_name,
            "confidence": round(confidence, 3),
            "bmi_estimate": bmi_estimate,
        }

    except Exception:
        traceback.print_exc()
        return {
            "error": "prediction_failed",
            "message": "ไม่สามารถประเมิน BMI จากภาพนี้ได้"
        }
