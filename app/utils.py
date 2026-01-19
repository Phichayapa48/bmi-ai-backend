import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# =========================
# Face detector
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_and_crop_face(pil_img):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    return Image.fromarray(face)

# =========================
# Preprocess (ตรงตอน train)
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ✅ ตรง train
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(face_img):
    return transform(face_img).unsqueeze(0)
