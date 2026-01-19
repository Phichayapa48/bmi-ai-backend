import numpy as np
from PIL import Image
import mediapipe as mp

mp_face = mp.solutions.face_detection

def detect_and_crop_face(
    pil_image: Image.Image,
    margin: float = 0.3
) -> Image.Image:

    img = np.array(pil_image.convert("RGB"))
    h, w, _ = img.shape

    with mp_face.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    ) as detector:

        results = detector.process(img)

        if not results.detections:
            raise ValueError("ไม่พบใบหน้าในภาพ")

        if len(results.detections) > 1:
            raise ValueError("พบหลายใบหน้าในภาพ")

        box = results.detections[0].location_data.relative_bounding_box

        cx = (box.xmin + box.width / 2) * w
        cy = (box.ymin + box.height / 2) * h

        size = max(box.width * w, box.height * h)
        size = size * (1 + margin)

        x1 = int(cx - size / 2)
        y1 = int(cy - size / 2)
        x2 = int(cx + size / 2)
        y2 = int(cy + size / 2)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        face = img[y1:y2, x1:x2]

        if face.size == 0:
            raise ValueError("crop ใบหน้าล้มเหลว")

        return Image.fromarray(face).convert("RGB")
