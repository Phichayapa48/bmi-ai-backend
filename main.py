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
# LOAD MODEL
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
# THRESHOLD
# =========================
CONF_THRESHOLD = 0.65
MARGIN_THRESHOLD = 0.15   # กันโมเดลลังเล

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
        # 🔒 กันไฟล์ไม่ใช่รูป
        if not file.content_type.startswith("image/"):
            return {"ok": False, "error": "invalid_file"}

        image = Image.open(io.BytesIO(await file.read())).convert("RGB")

        # =========================
        # 1️⃣ Face gate
        # =========================
        face_image, has_face = detect_and_crop_face(image)

        if not has_face:
            return decide(
                cls_name=None,
                confidence=0.0,
                face_ok=False,
                quality_ok=True,
                debug={"stage": "no_face"}
            )

        # 👇 กันรูปหมา / หน้าเล็กเกิน
        face_area_ratio = (face_image.width * face_image.height) / (image.width * image.height)

        if face_area_ratio < 0.05:
            return decide(
                cls_name=None,
                confidence=0.0,
                face_ok=False,
                quality_ok=True,
                debug={
                    "stage": "face_too_small",
                    "face_area_ratio": face_area_ratio
                }
            )

        # =========================
        # 2️⃣ Quality gate
        # =========================
        quality_ok, quality_score = quality_check(face_image)

        if not quality_ok:
            return decide(
                cls_name=None,
                confidence=0.0,
                face_ok=True,
                quality_ok=False,
                debug={
                    "stage": "low_quality",
                    "quality_score": quality_score
                }
            )

        # =========================
        # 3️⃣ Preprocess
        # =========================
        x = preprocess_image(face_image)
        x = x.to(next(model.parameters()).device)

        # =========================
        # 4️⃣ Predict
        # =========================
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]

        cls_idx = int(probs.argmax().item())
        confidence = float(probs[cls_idx])
        cls_name = BMI_LABELS[cls_idx]

        # =========================
        # 5️⃣ Margin check
        # =========================
        sorted_probs = torch.sort(probs, descending=True).values
        margin = float(sorted_probs[0] - sorted_probs[1])

        # =========================
        # 6️⃣ Debug info
        # =========================
        debug_info = {
            "logits": logits[0].tolist(),
            "probs": probs.tolist(),
            "pred_idx": cls_idx,
            "pred_label": cls_name,
            "confidence": confidence,
            "margin": margin,
            "input_stats": {
                "mean": float(x.mean()),
                "std": float(x.std()),
                "min": float(x.min()),
                "max": float(x.max())
            }
        }

        # =========================
        # 7️⃣ Confidence + margin gate
        # =========================
        if confidence < CONF_THRESHOLD or margin < MARGIN_THRESHOLD:
            return decide(
                cls_name=None,
                confidence=confidence,
                face_ok=True,
                quality_ok=True,
                debug={**debug_info, "stage": "low_conf_or_uncertain"}
            )

        # =========================
        # ✅ ผ่านทุก gate
        # =========================
        return decide(
            cls_name=cls_name,
            confidence=confidence,
            face_ok=True,
            quality_ok=True,
            debug={**debug_info, "stage": "ok"}
        )

    except Exception as e:
        print("❌ ERROR:", e)
        return {
            "ok": False,
            "error": "prediction_failed",
            "message": "ไม่สามารถประเมินจากภาพนี้ได้งับ"
        }
