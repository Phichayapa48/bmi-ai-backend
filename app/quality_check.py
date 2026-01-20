import numpy as np
import cv2
from PIL import Image


def quality_check(image: Image.Image):
    """
    ตรวจคุณภาพภาพแบบเบา ๆ (fast & practical)
    ใช้เป็น gate ก่อนส่งเข้า model
    """

    img = np.array(image)
    h, w, _ = img.shape

    # =========================
    # 1️⃣ Size gate
    # =========================
    if h < 120 or w < 120:
        return False, "ภาพมีขนาดเล็กเกินไป"

    # =========================
    # 2️⃣ Aspect ratio gate
    # (กันรูปยาว/กว้างผิดปกติ)
    # =========================
    ratio = w / h
    if ratio < 0.5 or ratio > 2.0:
        return False, "สัดส่วนภาพไม่เหมาะสม"

    # =========================
    # 3️⃣ Brightness gate
    # =========================
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)

    if brightness < 50:
        return False, "ภาพมืดเกินไป"

    if brightness > 220:
        return False, "ภาพสว่างเกินไป"

    # =========================
    # 4️⃣ Sharpness gate (Laplacian)
    # =========================
    gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.Laplacian(gray_norm, cv2.CV_64F).var()

    if blur < 25:
        return False, "ภาพไม่คมชัด"

    # =========================
    # PASS
    # =========================
    return True, "ผ่าน"
