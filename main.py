from fastapi import FastAPI, UploadFile, File
from PIL import Image, ImageOps, UnidentifiedImageError
import io
import torch

from app.model import get_model
from app.face_utils import detect_and_crop_face
from app.utils import preprocess_image
from app.quality_check import quality_check
from app.decision import decide


# =========================================================
# APP
# =========================================================
app = FastAPI(
    title="BMI AI Backend",
    version="1.1.0"
)


# =========================================================
# LOAD MODEL
# =========================================================
model = get_model()
model.eval()

print("✅ BMI model loaded successfully")


# =========================================================
# CLASS CONFIG
# ต้องตรงกับตอน Train
# =========================================================
BMI_LABELS = {
    0: "under",
    1: "normal",
    2: "over"
}


# =========================================================
# THRESHOLD
# =========================================================

# ความมั่นใจขั้นต่ำของโมเดล
CONF_THRESHOLD = 0.65

# ความต่างระหว่างอันดับ 1 กับอันดับ 2
MARGIN_THRESHOLD = 0.15

# เดิม 0.05
# ลดเหลือ 0.02 เพื่อไม่ให้ปฏิเสธรูปที่ใบหน้าเล็กเกินความจำเป็น
MIN_FACE_AREA_RATIO = 0.02


# =========================================================
# HEALTH CHECK
# =========================================================
@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "BMI AI Backend",
        "version": "1.1.0"
    }


# =========================================================
# PREDICT
# =========================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        # =====================================================
        # 0️⃣ CHECK FILE TYPE
        # =====================================================
        content_type = file.content_type or ""

        print("\n======================================")
        print("📥 NEW IMAGE REQUEST")
        print("FILE NAME:", file.filename)
        print("CONTENT TYPE:", content_type)

        if not content_type.startswith("image/"):

            print("❌ INVALID FILE TYPE")

            return {
                "ok": False,
                "error": "invalid_file",
                "message": "กรุณาส่งไฟล์รูปภาพค่ะ"
            }


        # =====================================================
        # READ IMAGE
        # =====================================================
        file_bytes = await file.read()

        print("IMAGE BYTES:", len(file_bytes))

        if not file_bytes:

            print("❌ EMPTY IMAGE")

            return {
                "ok": False,
                "error": "empty_file",
                "message": "ไม่พบข้อมูลรูปภาพ กรุณาส่งรูปใหม่ค่ะ"
            }


        # =====================================================
        # OPEN IMAGE
        # =====================================================
        try:

            image = Image.open(
                io.BytesIO(file_bytes)
            )

            # แก้ปัญหารูปจากมือถือหมุนตาม EXIF
            image = ImageOps.exif_transpose(image)

            # บังคับ RGB
            image = image.convert("RGB")

        except UnidentifiedImageError:

            print("❌ PIL CANNOT READ IMAGE")

            return {
                "ok": False,
                "error": "invalid_image",
                "message": "ระบบไม่สามารถอ่านไฟล์รูปภาพนี้ได้ค่ะ"
            }


        print(
            "🖼 ORIGINAL IMAGE SIZE:",
            image.width,
            "x",
            image.height
        )


        # =====================================================
        # 1️⃣ FACE GATE
        # =====================================================
        face_image, has_face = detect_and_crop_face(image)

        print("👤 HAS FACE:", has_face)


        # =====================================================
        # NO FACE
        # =====================================================
        if not has_face:

            print("❌ FACE NOT FOUND")

            return decide(
                cls_name=None,
                confidence=0.0,
                face_ok=False,
                quality_ok=True,
                debug={
                    "stage": "no_face",
                    "image_width": image.width,
                    "image_height": image.height
                }
            )


        # =====================================================
        # FACE FOUND
        # =====================================================
        print(
            "✅ FACE SIZE:",
            face_image.width,
            "x",
            face_image.height
        )


        # =====================================================
        # FACE AREA RATIO
        # =====================================================
        original_area = (
            image.width *
            image.height
        )

        face_area = (
            face_image.width *
            face_image.height
        )

        if original_area <= 0:

            print("❌ INVALID IMAGE SIZE")

            return {
                "ok": False,
                "error": "invalid_image_size"
            }


        face_area_ratio = (
            face_area /
            original_area
        )


        print(
            "📐 FACE AREA RATIO:",
            round(face_area_ratio, 4)
        )


        # =====================================================
        # FACE TOO SMALL
        # =====================================================
        if face_area_ratio < MIN_FACE_AREA_RATIO:

            print(
                "⚠️ FACE TOO SMALL:",
                face_area_ratio
            )

            return decide(
                cls_name=None,
                confidence=0.0,

                # สำคัญ:
                # เราตรวจเจอหน้าแล้ว
                # จึงไม่ควรบอกว่า face_ok=False
                face_ok=True,

                quality_ok=False,

                debug={
                    "stage": "face_too_small",
                    "face_area_ratio": float(
                        face_area_ratio
                    ),
                    "minimum_ratio": MIN_FACE_AREA_RATIO,
                    "original_size": [
                        image.width,
                        image.height
                    ],
                    "face_size": [
                        face_image.width,
                        face_image.height
                    ]
                }
            )


        # =====================================================
        # 2️⃣ QUALITY GATE
        # =====================================================
        quality_ok, quality_score = quality_check(
            face_image
        )

        print(
            "🔍 QUALITY OK:",
            quality_ok
        )

        print(
            "🔍 QUALITY SCORE:",
            quality_score
        )


        if not quality_ok:

            print("❌ LOW QUALITY IMAGE")

            return decide(
                cls_name=None,
                confidence=0.0,
                face_ok=True,
                quality_ok=False,
                debug={
                    "stage": "low_quality",
                    "quality_score": float(
                        quality_score
                    )
                }
            )


        # =====================================================
        # 3️⃣ PREPROCESS
        # =====================================================
        x = preprocess_image(
            face_image
        )


        # ส่ง tensor ไป device เดียวกับ model
        device = next(
            model.parameters()
        ).device

        x = x.to(device)


        print(
            "🧠 MODEL DEVICE:",
            device
        )

        print(
            "🧠 INPUT SHAPE:",
            tuple(x.shape)
        )


        # =====================================================
        # 4️⃣ MODEL PREDICT
        # =====================================================
        with torch.no_grad():

            logits = model(x)

            probs = torch.softmax(
                logits,
                dim=1
            )[0]


        # =====================================================
        # CLASS RESULT
        # =====================================================
        cls_idx = int(
            probs.argmax().item()
        )


        # กัน index ที่ไม่รู้จัก
        if cls_idx not in BMI_LABELS:

            print(
                "❌ UNKNOWN CLASS:",
                cls_idx
            )

            return {
                "ok": False,
                "error": "unknown_class",
                "message": "โมเดลส่งผลลัพธ์ที่ไม่ถูกต้อง"
            }


        cls_name = BMI_LABELS[
            cls_idx
        ]


        confidence = float(
            probs[cls_idx].item()
        )


        # =====================================================
        # 5️⃣ MARGIN CHECK
        # =====================================================
        sorted_probs = torch.sort(
            probs,
            descending=True
        ).values


        # ป้องกันกรณี class < 2
        if len(sorted_probs) >= 2:

            margin = float(
                (
                    sorted_probs[0] -
                    sorted_probs[1]
                ).item()
            )

        else:

            margin = confidence


        # =====================================================
        # PRINT PREDICTION
        # =====================================================
        print("------------------------------")
        print("🎯 PREDICTED CLASS:", cls_name)

        print(
            "🎯 CONFIDENCE:",
            round(confidence, 4)
        )

        print(
            "🎯 MARGIN:",
            round(margin, 4)
        )

        print(
            "🎯 PROBS:",
            probs.detach().cpu().tolist()
        )

        print("------------------------------")


        # =====================================================
        # 6️⃣ DEBUG INFO
        # =====================================================
        debug_info = {

            "stage": "prediction",

            "image_size": [
                image.width,
                image.height
            ],

            "face_size": [
                face_image.width,
                face_image.height
            ],

            "face_area_ratio":
                float(face_area_ratio),

            "quality_score":
                float(quality_score),

            "logits":
                logits[0]
                .detach()
                .cpu()
                .tolist(),

            "probs":
                probs
                .detach()
                .cpu()
                .tolist(),

            "pred_idx":
                cls_idx,

            "pred_label":
                cls_name,

            "confidence":
                confidence,

            "margin":
                margin,

            "input_stats": {

                "mean":
                    float(
                        x.mean().item()
                    ),

                "std":
                    float(
                        x.std().item()
                    ),

                "min":
                    float(
                        x.min().item()
                    ),

                "max":
                    float(
                        x.max().item()
                    )
            }
        }


        # =====================================================
        # 7️⃣ CONFIDENCE + MARGIN GATE
        # =====================================================
        if (
            confidence < CONF_THRESHOLD
            or
            margin < MARGIN_THRESHOLD
        ):

            print("⚠️ LOW CONFIDENCE / UNCERTAIN")

            print(
                "confidence:",
                confidence
            )

            print(
                "margin:",
                margin
            )

            return decide(
                cls_name=None,
                confidence=confidence,
                face_ok=True,
                quality_ok=True,
                debug={
                    **debug_info,
                    "stage":
                        "low_conf_or_uncertain"
                }
            )


        # =====================================================
        # ✅ PASSED ALL GATES
        # =====================================================
        print("✅ PREDICTION SUCCESS")

        print(
            "RESULT:",
            cls_name
        )

        print(
            "CONFIDENCE:",
            confidence
        )

        print("======================================\n")


        return decide(
            cls_name=cls_name,
            confidence=confidence,
            face_ok=True,
            quality_ok=True,
            debug={
                **debug_info,
                "stage": "ok"
            }
        )


    # =========================================================
    # ERROR
    # =========================================================
    except Exception as e:

        print("\n❌ PREDICTION ERROR")
        print(
            "TYPE:",
            type(e).__name__
        )

        print(
            "MESSAGE:",
            str(e)
        )

        print("======================================\n")


        return {
            "ok": False,
            "error": "prediction_failed",
            "message":
                "ไม่สามารถประเมินจากภาพนี้ได้ กรุณาลองส่งรูปใหม่อีกครั้ง"
        }
