import cv2
import numpy as np
from PIL import Image

# โหลด Haar Cascade (มากับ opencv อยู่แล้ว)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_and_crop_face(pil_image: Image.Image) -> Image.Image:
    # PIL → numpy
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        raise ValueError("ไม่พบใบหน้าในภาพ")

    if len(faces) > 1:
        raise ValueError("พบหลายใบหน้าในภาพ")

    x, y, w, h = faces[0]

    # กันหลุดขอบ
    h_img, w_img, _ = img.shape
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)

    face = img[y1:y2, x1:x2]

    if face.size == 0:
        raise ValueError("crop ใบหน้าล้มเหลว")

    return Image.fromarray(face).convert("RGB")
