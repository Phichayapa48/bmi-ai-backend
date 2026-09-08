import cv2
import numpy as np
from PIL import Image

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_and_crop_face(image: Image.Image):
    image = image.convert("RGB")

    # รูปต้นฉบับ
    original = np.array(image)
    orig_h, orig_w = original.shape[:2]

    # ใช้รูปย่อเฉพาะตอน detect เพื่อให้เร็ว
    scale = 1.0
    detect_img = original

    if max(orig_h, orig_w) > 1000:
        scale = 1000.0 / max(orig_h, orig_w)

        detect_img = cv2.resize(
            original,
            (
                int(orig_w * scale),
                int(orig_h * scale)
            )
        )

    gray = cv2.cvtColor(
        detect_img,
        cv2.COLOR_RGB2GRAY
    )

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return image, False

    # เลือกหน้าที่ใหญ่ที่สุด
    x, y, w, h = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    # แปลงตำแหน่งกลับไปขนาดรูปต้นฉบับ
    x = int(x / scale)
    y = int(y / scale)
    w = int(w / scale)
    h = int(h / scale)

    # เพิ่มพื้นที่รอบใบหน้า
    pad = int(0.25 * w)

    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)

    x2 = min(x + w + pad, orig_w)
    y2 = min(y + h + pad, orig_h)

    face = original[y1:y2, x1:x2]

    if face.size == 0:
        return image, False

    return Image.fromarray(face), True
