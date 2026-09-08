import cv2
import numpy as np

from PIL import Image


# =========================================================
# LOAD HAAR CASCADE
# =========================================================
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)


# =========================================================
# DETECT + CROP FACE
# =========================================================
def detect_and_crop_face(image: Image.Image):

    # บังคับ RGB
    image = image.convert("RGB")

    # รูปต้นฉบับ
    original = np.array(image)

    orig_h, orig_w = original.shape[:2]


    print(
        "🔎 FACE DETECTOR ORIGINAL:",
        orig_w,
        "x",
        orig_h
    )


    # =====================================================
    # RESIZE เฉพาะสำหรับ DETECT
    # =====================================================
    detect_img = original.copy()

    scale = 1.0


    if max(orig_h, orig_w) > 1000:

        scale = (
            1000.0 /
            max(orig_h, orig_w)
        )


        new_w = max(
            1,
            int(orig_w * scale)
        )

        new_h = max(
            1,
            int(orig_h * scale)
        )


        detect_img = cv2.resize(
            original,
            (
                new_w,
                new_h
            ),
            interpolation=cv2.INTER_AREA
        )


    print(
        "🔎 DETECT SCALE:",
        scale
    )

    print(
        "🔎 DETECT IMAGE:",
        detect_img.shape[1],
        "x",
        detect_img.shape[0]
    )


    # =====================================================
    # RGB → GRAYSCALE
    # =====================================================
    gray = cv2.cvtColor(
        detect_img,
        cv2.COLOR_RGB2GRAY
    )


    # ช่วยเพิ่ม contrast
    gray = cv2.equalizeHist(
        gray
    )


    # =====================================================
    # DETECT FACE
    # =====================================================
    faces = FACE_CASCADE.detectMultiScale(

        gray,

        scaleFactor=1.1,

        minNeighbors=4,

        minSize=(40, 40)
    )


    print(
        "🔎 FACES FOUND:",
        len(faces)
    )


    # =====================================================
    # NO FACE
    # =====================================================
    if len(faces) == 0:

        return image, False


    # =====================================================
    # เลือกหน้าที่ใหญ่ที่สุด
    # =====================================================
    x, y, w, h = max(
        faces,
        key=lambda f:
            f[2] * f[3]
    )


    print(
        "🔎 DETECTED FACE:",
        x,
        y,
        w,
        h
    )


    # =====================================================
    # แปลงพิกัดกลับไปยังรูปต้นฉบับ
    # =====================================================
    if scale != 1.0:

        x = int(
            x / scale
        )

        y = int(
            y / scale
        )

        w = int(
            w / scale
        )

        h = int(
            h / scale
        )


    # =====================================================
    # เพิ่มพื้นที่รอบใบหน้า
    # =====================================================
    pad_x = int(
        w * 0.20
    )

    pad_y = int(
        h * 0.25
    )


    x1 = max(
        0,
        x - pad_x
    )

    y1 = max(
        0,
        y - pad_y
    )

    x2 = min(
        orig_w,
        x + w + pad_x
    )

    y2 = min(
        orig_h,
        y + h + pad_y
    )


    # =====================================================
    # CROP จากรูปต้นฉบับ
    # =====================================================
    face = original[
        y1:y2,
        x1:x2
    ]


    if face.size == 0:

        print(
            "❌ EMPTY FACE CROP"
        )

        return image, False


    face_image = Image.fromarray(
        face
    )


    print(
        "✅ FACE CROP:",
        face_image.width,
        "x",
        face_image.height
    )


    return face_image, True
