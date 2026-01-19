import numpy as np

def preprocess(pil_face):
    img = pil_face.resize((224, 224))
    img = np.array(img) / 255.0
    return img
