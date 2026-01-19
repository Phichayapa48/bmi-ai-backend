import cv2
import numpy as np

def quality_check(pil_img):
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur < 80:
        return False, "ภาพเบลอเกินไป"

    contrast = gray.std()
    if contrast < 30:
        return False, "ภาพคอนทราสต์ต่ำ"

    h, w = gray.shape
    if h < 300 or w < 300:
        return False, "ความละเอียดต่ำเกินไป"

    return True, None
