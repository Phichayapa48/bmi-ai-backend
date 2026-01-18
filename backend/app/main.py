from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import torch
import cv2
import numpy as np

from app.model import get_model
from app.utils import preprocess_image

app = FastAPI(title="BMI Face AI API")

# -------------------------
# Load face detector
# -------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------------
# Startup
# -------------------------
@app.on_event("startup")
def load_model_on_startup():
    get_model()

@app.get("/")
def root():
    return {"status": "ok"}

# -------------------------
# BMI Category
# -------------------------
def bmi_category(bmi: float):
    if bmi < 18.5:
        return "underweight", "คุณมีน้ำหนักต่ำกว่าเกณฑ์ 🥺"
    elif bmi < 23:
        return "normal", "คุณมีน้ำหนักอยู่ในเกณฑ์ปกติ 👍"
    elif bmi < 25:
        return "overweight", "คุณมีน้ำหนักเกินเล็กน้อย 😅"
    else:
        return "obese", "คุณมีน้ำหนักเกินเกณฑ์มาก ⚠️"

# -------------------------
# Predict
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ---------- Read image ----------
    try:
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # ---------- Face Detection ----------
    open_cv_image = np.array(pil_image)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return {
            "bmi": None,
            "category": "no_face",
            "confidence": 0.0,
            "message": "ไม่พบใบหน้าคนในภาพ กรุณาถ่ายรูปใบหน้าให้ชัดเจนอีกครั้งนะคะ 🙂"
        }

    # ---------- Select largest face ----------
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_img = pil_image.crop((x, y, x + w, y + h))
    face_img = face_img.resize((224, 224))

    # ---------- Preprocess ----------
    x_tensor = preprocess_image(face_img)

    # ---------- Predict BMI ----------
    model = get_model()
    with torch.no_grad():
        pred = model(x)
        bmi = float(pred.item())

    # ---------- Confidence (Regression heuristic) ----------
    # สมมุติ error ±2 BMI = confidence ต่ำ
    error_margin = 2.0
    confidence = max(0.0, 1.0 - abs(bmi_pred - round(bmi_pred)) / error_margin)

    # ---------- Uncertain ----------
    if confidence < 0.5 or bmi_pred < 10 or bmi_pred > 45:
        return {
            "bmi": round(bmi_pred, 1),
            "category": "uncertain",
            "confidence": round(confidence, 2),
            "message": "ระบบยังไม่มั่นใจในผลลัพธ์ กรุณาลองถ่ายรูปใหม่อีกครั้งนะคะ 🙂"
        }

    # ---------- Normal ----------
    category, message = bmi_category(bmi_pred)

    return {
        "bmi": round(bmi_pred, 1),
        "category": category,
        "confidence": round(confidence, 2),
        "message": message
    }
