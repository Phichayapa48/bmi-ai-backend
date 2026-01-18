from torchvision import transforms
import torch

IMAGE_SIZE = 224

_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image):
    x = _transform(image)
    x = x.unsqueeze(0)          # [1,3,224,224]
    x = x.to(dtype=torch.float32)
    return x
