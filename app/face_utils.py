import numpy as np
from PIL import Image
import mediapipe as mp

mp_face = mp.solutions.face_detection


def detect_and_crop_face(pil_image: Image.Image) -> Image.Image:
    # PIL → numpy (RGB)
    img = np.array(pil_image.convert("RGB"))
    h, w, _ = img.shape

    with mp_face.FaceDetection(
        model_selection=0,            # เร็ว + เหมาะกับ CPU (Render)
        min_detection_confidence=0.5
    ) as detector:
        results = detector.process(img)

        if not results.detections:
            raise ValueError("ไม่พบใบหน้าในภาพ")

        if len(results.detections) > 1:
            raise ValueError("พบหลายใบหน้าในภาพ")

        det = results.detections[0]
        box = det.location_data.relative_bounding_box

        # แปลงเป็น pixel
        x1 = int(box.xmin * w)
        y1 = int(box.ymin * h)
        x2 = int((box.xmin + box.width) * w)
        y2 = int((box.ymin + box.height) * h)

        # กันค่าหลุดขอบ
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        face = img[y1:y2, x1:x2]

        if face.size == 0:
            raise ValueError("crop ใบหน้าล้มเหลว")

        # ขยายขอบนิดหน่อย (กันหน้าชิดเกิน)
        pad = int(0.1 * max(face.shape[0], face.shape[1]))
        face = Image.fromarray(face).convert("RGB")

        return face
