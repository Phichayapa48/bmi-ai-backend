from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import traceback
import torch

from app.model import get_model
from app.utils import preprocess_image
from app.face_utils import detect_and_crop_face
from app.quality_check import quality_check

# =========================
# THAI LABEL
# =========================
BMI_STATUS_TH = {
    "under": "ต่ำกว่าเกณฑ์",
    "normal": "สมส่วน",
    "over": "สูงกว่าเกณฑ์"
}

# =========================
# CLASS CONFIG (ต้องตรงตอน train)
# =========================
BMI_LABELS = {
    0: ("under", 17.5),
    1: ("normal", 22.0),
    2: ("over", 27.5)
}

app = FastAPI()

# =========================
# GLOBAL MODEL
# =========================
model = None

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

        # 2️⃣ Quality check (เตือนเฉย ๆ ไม่ตัด)
        ok, reason = quality_check(image)
        if not ok:
            print(f"⚠️ Quality warning: {reason}")

        # 3️⃣ Detect & crop face
        face_image, face_found = detect_and_crop_face(image)

        if face_found:
            print("🙂 พบใบหน้า → ใช้ภาพที่ crop")
        else:
            print("⚠️ ไม่พบใบหน้า → ใช้ทั้งภาพ")

        # 4️⃣ Preprocess (224x224 ตรง train)
        x = preprocess_image(face_image)
        x = x.to(next(model.parameters()).device)

        # 5️⃣ Predict (ใช้ผลโมเดลล้วน ๆ)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            cls_idx = int(probs.argmax(dim=1).item())
            cls_name, _ = BMI_LABELS[cls_idx]
            confidence = float(probs[0, cls_idx])

        # 6️⃣ Response (สำหรับผู้ใช้)
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
