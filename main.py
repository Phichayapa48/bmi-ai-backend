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
    0: "under",
    1: "normal",
    2: "over"
}

BMI_STATUS_TH = {
    "under": "ต่ำกว่าเกณฑ์",
    "normal": "สมส่วน",
    "over": "สูงกว่าเกณฑ์"
}

# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "BMI AI Backend"
    }

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

        # ❗ กันรูปเล็กเกิน
        if image.width < 80 or image.height < 80:
            return {
                "error": "image_too_small",
                "message": "ภาพมีขนาดเล็กเกินไป กรุณาถ่ายใหม่"
            }

        # 2️⃣ Quality gate
        quality_ok, reason = quality_check(image)
        if not quality_ok:
            return {
                "error": "bad_quality",
                "message": "ภาพไม่ชัดหรือแสงไม่เพียงพอ กรุณาถ่ายใหม่"
            }

        # 3️⃣ Face gate (สำคัญมาก)
        face_image, has_face = detect_and_crop_face(image)
        if not has_face:
            return {
                "error": "no_face",
                "message": "ไม่พบใบหน้าที่ชัดเจน กรุณาถ่ายภาพใบหน้าตรง ไม่ใส่หน้ากาก"
            }

        # 4️⃣ Preprocess
        x = preprocess_image(face_image)
        x = x.to(next(model.parameters()).device)

        # 5️⃣ Predict
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            cls_idx = int(probs.argmax(dim=1).item())
            cls_name = BMI_LABELS[cls_idx]
            confidence = float(probs[0, cls_idx])

        # 🔍 debug log (เอาไว้ดู bias)
        print("🧠 PRED:", cls_name, confidence)

        # 6️⃣ Threshold แยกตามคลาส (แก้ under bias)
        class_thresholds = {
            "under": 0.60,
            "normal": 0.45,
            "over": 0.50
        }

        threshold = class_thresholds.get(cls_name, 0.5)

        if confidence < threshold:
            return {
                "error": "low_confidence",
                "message": "ไม่สามารถประเมินได้อย่างมั่นใจ กรุณาถ่ายภาพใหม่"
            }

        # 7️⃣ Final response
        return {
            "status": BMI_STATUS_TH[cls_name],
            "confidence": round(confidence, 3)
        }

    except Exception:
        traceback.print_exc()
        return {
            "error": "prediction_failed",
            "message": "ไม่สามารถประเมิน BMI จากภาพนี้ได้"
        }
