def decide(
    cls_name,
    confidence,
    face_ok=True,
    quality_ok=True
):
    """
    cls_name: 'under' | 'normal' | 'over'
    confidence: float (0-1)
    face_ok: ผ่าน face detection หรือไม่
    quality_ok: ภาพมีคุณภาพพอหรือไม่
    """

    # ❌ ไม่ผ่าน gate
    if not face_ok:
        return {
            "ok": False,
            "message": "ไม่พบใบหน้าที่ชัดเจน กรุณาถ่ายภาพใหม่"
        }

    if not quality_ok:
        return {
            "ok": False,
            "message": "ภาพไม่ชัดหรือมืดเกินไป กรุณาถ่ายใหม่"
        }

    # 🔧 threshold แยกตามคลาส (สำคัญมาก)
    class_thresholds = {
        "under": 0.55,
        "normal": 0.45,
        "over": 0.50
    }

    threshold = class_thresholds.get(cls_name, 0.5)

    if confidence < threshold:
        return {
            "ok": False,
            "message": "ผลลัพธ์ยังไม่มั่นใจ กรุณาถ่ายภาพใหม่"
        }

    return {
        "ok": True,
        "class": cls_name,
        "confidence": round(float(confidence), 3)
    }
