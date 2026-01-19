from torchvision import transforms
import torch

def smart_crop(image):
    """
    crop กลางภาพ + bias ลงล่าง
    ได้หน้า + คอ + ไหล่
    """
    w, h = image.size
    crop_size = int(min(w, h) * 0.85)

    cx = w // 2
    cy = int(h * 0.45)

    left = max(cx - crop_size // 2, 0)
    top = max(cy - crop_size // 2, 0)
    right = min(left + crop_size, w)
    bottom = min(top + crop_size, h)

    return image.crop((left, top, right, bottom))


def preprocess_image(image):
    image = smart_crop(image)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ✅ ตามที่เพื่อนบอก + ตรงตอน train
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    x = transform(image).unsqueeze(0)
    return x
