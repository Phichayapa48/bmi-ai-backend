import torch
from torchvision import transforms

def preprocess_image(pil_image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # 🔥 ตรงกับตอน train
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet
            std=[0.229, 0.224, 0.225]
        )
    ])

    x = transform(pil_image)
    x = x.unsqueeze(0)  # (1, 3, 224, 224)

    return x
