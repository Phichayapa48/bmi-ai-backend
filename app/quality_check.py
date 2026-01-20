import numpy as np
import cv2
from PIL import Image

def quality_check(image: Image.Image):
    img = np.array(image)

    if img.ndim != 3:
        return False, "รูปไม่ถูกต้อง"

    h, w, _ = img.shape
    if h < 120 or w < 120:
        return False, "ใบหน้าเล็กเกินไป"

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    if blur_score < 30:
        return False, "ภาพเบลอเกินไป"

    return True, "ผ่าน"
