from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import traceback
import torch

from app.model import get_model
from app.utils import preprocess_image
from app.face_utils import detect_and_crop_face
from app.quality_check import quality_check

BMI_STATUS_TH = {
    "under": "ต่ำกว่าเกณฑ์",
    "normal": "สมส่วน",
    "over": "สูงกว่าเกณฑ์"
}

app = FastAPI()

# =========================
# GLOBAL MODEL
# =========================
model = None

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

        # 2️⃣ Quality check (soft warning)
        ok, reason = quality_check(image)
        if not ok:
            print(f"⚠️ Quality warning: {reason}")

        # 3️⃣ Detect face (soft logic)
        face_image, face_found = detect_and_crop_face(image)

        # ❗ ถ้าไม่เจอหน้าเลย + confidence ต่ำ → ค่อย reject
        # (ตอนนี้ยังให้โมเดลลองก่อน)
        x = preprocess_image(face_image)
        x = x.to(next(model.parameters()).device)

        # 4️⃣ Predict
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            cls_idx = int(probs.argmax(dim=1).item())
            confidence = float(probs[0, cls_idx])

        cls_name = BMI_LABELS.get(cls_idx)

        # 5️⃣ Hard reject เฉพาะกรณี "ไม่น่าใช่คนจริง ๆ"
        if not face_found and confidence < 0.55:
            return {
                "error": "no_clear_face",
                "message": "ไม่พบใบหน้าที่ชัดเจน กรุณาถ่ายภาพใบหน้าคนเท่านั้น"
            }

        # 6️⃣ Response (ฝั่งผู้ใช้)
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
