import cv2
import numpy as np
from PIL import Image

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_and_crop_face(image: Image.Image):
    image = image.convert("RGB")
    img = np.array(image)

    h, w, _ = img.shape
    if max(h, w) > 1000:
        scale = 1000 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return image, False

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    pad = int(0.25 * w)
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, img.shape[1])
    y2 = min(y + h + pad, img.shape[0])

    face = img[y1:y2, x1:x2]
    return Image.fromarray(face), True
