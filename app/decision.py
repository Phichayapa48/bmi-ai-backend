def decide(
    cls_name: str,
    confidence: float,
    face_ok: bool = True,
    quality_ok: bool = True
):
    """
    cls_name: 'under' | 'normal' | 'over'
    confidence: float (0-1)
    face_ok: ผ่าน face detection หรือไม่
    quality_ok: ภาพมีคุณภาพพอหรือไม่
    """

    # =========================
    # GATE: FACE
    # =========================
    if not face_ok:
        return {
            "ok": False,
            "error": "no_face",
            "message": "ไม่พบใบหน้าที่ชัดเจน กรุณาถ่ายภาพใบหน้าตรง"
        }

    # =========================
    # GATE: QUALITY
    # =========================
    if not quality_ok:
        return {
            "ok": False,
            "error": "bad_quality",
            "message": "ภาพไม่ชัดหรือแสงไม่เพียงพอ กรุณาถ่ายใหม่"
        }

    # =========================
    # THRESHOLD PER CLASS
    # (แก้ under bias)
    # =========================
    class_thresholds = {
        "under": 0.60,   # เข้มขึ้น
        "normal": 0.45,
        "over": 0.50
    }

    threshold = class_thresholds.get(cls_name, 0.5)

    if confidence < threshold:
        return {
            "ok": False,
            "error": "low_confidence",
            "message": "ผลลัพธ์ยังไม่มั่นใจ กรุณาถ่ายภาพใหม่"
        }

    # =========================
    # PASS
    # =========================
    return {
        "ok": True,
        "class": cls_name,
        "confidence": round(float(confidence), 3)
    }
