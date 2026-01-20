def decide(
    cls_name: str | None,
    confidence: float,
    face_ok: bool = True,
    quality_ok: bool = True
):
    """
    cls_name: 'under' | 'normal' | 'over' | None
    confidence: float (0-1)
    """

    # ❌ ไม่ผ่าน face gate
    if not face_ok:
        return {
            "ok": False,
            "error": "no_face",
            "message": "ไม่พบใบหน้าที่ชัดเจน กรุณาถ่ายภาพใบหน้าตรง"
        }

    # ❌ ไม่ผ่าน quality gate
    if not quality_ok:
        return {
            "ok": False,
            "error": "bad_quality",
            "message": "ภาพไม่ชัดหรือแสงไม่เพียงพอ กรุณาถ่ายใหม่"
        }

    # ❌ cls ผิดปกติ (กันพลาด)
    if cls_name not in {"under", "normal", "over"}:
        return {
            "ok": False,
            "error": "invalid_class",
            "message": "ไม่สามารถประเมินผลได้ กรุณาถ่ายภาพใหม่"
        }

    # ❌ confidence ผิดปกติ
    if not (0.0 <= confidence <= 1.0):
        return {
            "ok": False,
            "error": "invalid_confidence",
            "message": "ไม่สามารถประเมินได้อย่างมั่นใจ กรุณาถ่ายภาพใหม่"
        }

    # 🔧 Threshold แยกตามคลาส
    class_thresholds = {
        "under": 0.55,
        "normal": 0.45,
        "over": 0.50
    }

    threshold = class_thresholds[cls_name]

    # ❌ ไม่มั่นใจ
    if confidence < threshold:
        return {
            "ok": False,
            "error": "low_confidence",
            "message": "ไม่สามารถประเมินได้อย่างมั่นใจ กรุณาถ่ายภาพใหม่"
        }

    # ✅ ผ่านทั้งหมด
    return {
        "ok": True,
        "class": cls_name,
        "confidence": round(float(confidence), 3)
    }
