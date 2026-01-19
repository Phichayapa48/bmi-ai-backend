import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# =========================
# Face detector
# =========================
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_and_crop_face(image: Image.Image):
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.05,     # 👈 ผ่อน
        minNeighbors=2,       # 👈 ผ่อน
        minSize=(40, 40)      # 👈 สำคัญมาก
    )

    # ✅ ไม่เจอหน้า → ใช้ทั้งภาพ แต่บอกว่าไม่เจอ
    if len(faces) == 0:
        return image, False

    # เลือกหน้าที่ใหญ่ที่สุด
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    pad = int(0.2 * w)
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, img.shape[1])
    y2 = min(y + h + pad, img.shape[0])

    face = img[y1:y2, x1:x2]
    return Image.fromarray(face), True


# =========================
# Preprocess (ตรงตอน train)
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # ✅ ตรงตอน train
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image):
    return transform(image).unsqueeze(0)
