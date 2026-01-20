from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch

from app.model import get_model
from app.face_utils import detect_and_crop_face
from app.utils import preprocess_image
from app.quality_check import quality_check
from app.decision import decide

app = FastAPI()

# =========================
# LOAD MODEL (singleton)
# =========================
model = get_model()
model.eval()

# =========================
# CLASS CONFIG (ตรงตอน train)
# =========================
BMI_LABELS = {
    0: "under",
    1: "normal",
    2: "over"
}

# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def health():
    return {"status": "ok", "service": "BMI AI Backend"}

# =========================
# PREDICT
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1️⃣ Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 2️⃣ Face gate (สำคัญที่สุด)
        face_image, has_face = detect_and_crop_face(image)
        if not has_face:
            return decide(
                cls_name=None,
                confidence=0.0,
                face_ok=False,
                quality_ok=True
            )

        # 3️⃣ Quality gate (เช็กเฉพาะหน้า)
        quality_ok, _ = quality_check(face_image)
        if not quality_ok:
            return decide(
                cls_name=None,
                confidence=0.0,
                face_ok=True,
                quality_ok=False
            )

        # 4️⃣ Preprocess
        x = preprocess_image(face_image)
        x = x.to(next(model.parameters()).device)

        # 5️⃣ Predict
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            cls_idx = int(probs.argmax(dim=1).item())
            confidence = float(probs[0, cls_idx])
            cls_name = BMI_LABELS[cls_idx]

        # 6️⃣ Final decision (ศูนย์รวม logic)
        return decide(
            cls_name=cls_name,
            confidence=confidence,
            face_ok=True,
            quality_ok=True
        )

    except Exception as e:
        print("❌ ERROR:", e)
        return {
            "ok": False,
            "error": "prediction_failed",
            "message": "ไม่สามารถประเมิน BMI จากภาพนี้ได้"
        }
