import numpy as np
import cv2
from PIL import Image

def quality_check(image: Image.Image):
    img = np.array(image)
    h, w, _ = img.shape

    if h < 100 or w < 100:
        return False, "ภาพมีขนาดเล็กเกินไป"

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()

    if blur < 10:
        return False, "ภาพไม่คมชัด"

    return True, "ผ่าน"
