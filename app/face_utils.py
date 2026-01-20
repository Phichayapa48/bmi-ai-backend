import cv2
import numpy as np
from PIL import Image

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_and_crop_face(image: Image.Image):
    # บังคับ RGB
    image = image.convert("RGB")
    img = np.array(image)

    h, w, _ = img.shape

    # ❌ กันรูปเล็กเกิน (สำคัญ)
    if h < 120 or w < 120:
        return image, False

    # resize กันรูปใหญ่เกิน
    if max(h, w) > 1000:
        scale = 1000 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        h, w, _ = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return image, False

    # เลือกหน้าที่ใหญ่สุด
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])

    # ❌ guard: หน้าต้องใหญ่พอเมื่อเทียบกับภาพ
    face_ratio = (fw * fh) / (h * w)
    if face_ratio < 0.02:   # < 2% ถือว่าเล็กเกิน
        return image, False

    # padding รอบหน้า
    pad = int(0.25 * fw)
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + fw + pad, w)
    y2 = min(y + fh + pad, h)

    face = img[y1:y2, x1:x2]

    return Image.fromarray(face), True
