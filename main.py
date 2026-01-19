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
# CLASS CONFIG
# =========================
LABELS = ["under", "normal", "over"]

# ค่า BMI representative (ปรับได้)
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
        # =========================
        # 1️⃣ อ่านรูป
        # =========================
        image_bytes = await file.read()

        if len(image_bytes) == 0:
            return {
                "error": "empty_file",
                "message": "ไม่พบข้อมูลรูปภาพ"
            }

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # =========================
        # 2️⃣ Quality check (soft)
        # =========================
        ok, reason = quality_check(image)
        if not ok:
            # ❗️ไม่ reject ทิ้ง แค่ log เตือน
            print(f"⚠️ Quality warning: {reason}")

        # =========================
        # 3️⃣ detect + crop ใบหน้า
        # (ถ้าไม่เจอหน้า → ใช้รูปทั้งภาพ)
        # =========================
        face_image = detect_and_crop_face(image)

        # =========================
        # 4️⃣ preprocess
        # =========================
        x = preprocess_image(face_image)

        # =========================
        # 5️⃣ predict
        # =========================
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            cls_idx = int(probs.argmax(dim=1).item())
            cls_name = LABELS[cls_idx]
            confidence = float(probs[0, cls_idx])

        # =========================
        # 6️⃣ post-process
        # =========================
        bmi_estimate = BMI_MAP.get(cls_name, None)

        return {
            "category": cls_name,
            "confidence": round(confidence, 3),
            "bmi_estimate": bmi_estimate,
            "message": "success"
        }

    except ValueError as ve:
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
