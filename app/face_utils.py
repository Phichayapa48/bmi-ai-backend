import cv2
import numpy as np
from PIL import Image

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_and_crop_face(pil_image: Image.Image) -> Image.Image:
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        raise ValueError("ไม่พบใบหน้า")
    if len(faces) > 1:
        raise ValueError("พบหลายใบหน้า")

    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]

    return Image.fromarray(face)
