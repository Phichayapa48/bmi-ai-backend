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
# CLASS INFO
# -------------------------
CLASS_INFO = {
    0: ("underweight", "คุณมีน้ำหนักต่ำกว่าเกณฑ์ 🥺"),
    1: ("normal", "คุณมีน้ำหนักอยู่ในเกณฑ์ปกติ 👍"),
    2: ("overweight", "คุณมีน้ำหนักเกินเกณฑ์ 😅"),
}

# -------------------------
# Load face detector
# -------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

@app.on_event("startup")
def load_model_on_startup():
    get_model()

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ---------- อ่านรูป ----------
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
            "class_id": -1,
            "class_name": "no_face",
            "confidence": 0.0,
            "message": "ไม่พบใบหน้าคนในภาพ กรุณาถ่ายรูปใบหน้าให้ชัดเจนอีกครั้งนะคะ 🙂"
        }

    # ---------- เลือกหน้าใหญ่สุด ----------
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_img = pil_image.crop((x, y, x + w, y + h))

    # ---------- Resize เป็น 224x224 ----------
    face_img = face_img.resize((224, 224))

    # ---------- Preprocess ----------
    model = get_model()
    x_tensor = preprocess_image(face_img)

    # ---------- Predict ----------
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1)

        top2 = torch.topk(probs, k=2, dim=1)
        top1_prob = top2.values[0][0].item()
        top2_prob = top2.values[0][1].item()
        class_id = top2.indices[0][0].item()

        gap = top1_prob - top2_prob

    # ---------- กรณีไม่มั่นใจ ----------
    if gap < 0.15:
        return {
            "class_id": -1,
            "class_name": "uncertain",
            "confidence": round(top1_prob, 2),
            "message": "ระบบยังไม่มั่นใจในผลลัพธ์ กรุณาลองถ่ายรูปใหม่อีกครั้งนะคะ 🙂"
        }

    # ---------- กรณีปกติ ----------
    class_name, message = CLASS_INFO[class_id]

    return {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": round(top1_prob, 2),
        "message": message
    }
