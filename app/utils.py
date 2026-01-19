from torchvision import transforms
import torch
import numpy as np

def smart_crop(image):
    """
    crop กลางภาพ + bias ลงล่าง
    ได้หน้า + คอ + ไหล่แน่นอน
    """
    w, h = image.size

    crop_size = int(min(w, h) * 0.85)

    cx = w // 2
    cy = int(h * 0.45)  # bias ลงล่าง 🔥

    left = max(cx - crop_size // 2, 0)
    top = max(cy - crop_size // 2, 0)
    right = min(left + crop_size, w)
    bottom = min(top + crop_size, h)

    return image.crop((left, top, right, bottom))
