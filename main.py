from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import traceback
import torch

from app.model import get_model
from app.utils import preprocess_image
from app.face_utils import detect_and_crop_face
from app.quality_check import quality_check
from app.decision import decide

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

        # 2️⃣ Quality check (soft check)
        ok, reason = quality_check(image)
        if not ok:
            print(f"⚠️ Quality warning: {reason}")

        # 3️⃣ Detect face (ไม่บังคับ แต่ลดความมั่นใจ)
        face_image, has_face = detect_and_crop_face(image)

        # 4️⃣ Preprocess (224x224)
        x = preprocess_image(face_image)
        x = x.to(next(model.parameters()).device)

        # 5️⃣ Predict
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            cls_idx = int(probs.argmax(dim=1).item())
            cls_name = BMI_LABELS[cls_idx]
            confidence = float(probs[0, cls_idx])

        # 6️⃣ ถ้าไม่เจอหน้า → ลด confidence
        if not has_face:
            confidence *= 0.7

        # 7️⃣ Decision layer (ตัดสินใจจริง)
        decision = decide(cls_name, confidence)

        if not decision["ok"]:
            return {
                "error": "low_confidence",
                "message": decision["message"]
            }

        # 8️⃣ Final response
        return {
            "status": BMI_STATUS_TH[decision["class"]],
            "confidence": decision["confidence"]
        }

    except Exception:
        traceback.print_exc()
        return {
            "error": "prediction_failed",
            "message": "ไม่สามารถประเมิน BMI จากภาพนี้ได้"
        }
