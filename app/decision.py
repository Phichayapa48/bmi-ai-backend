def decide(
    cls_name: str,
    confidence: float,
    face_ok: bool = True,
    quality_ok: bool = True
):
    """
    cls_name: 'under' | 'normal' | 'over'
    confidence: float (0-1)
    """

    if not face_ok:
        return {
            "ok": False,
            "error": "no_face",
            "message": "ไม่พบใบหน้าที่ชัดเจน กรุณาถ่ายภาพใบหน้าตรง"
        }

    if not quality_ok:
        return {
            "ok": False,
            "error": "bad_quality",
            "message": "ภาพไม่ชัดหรือแสงไม่เพียงพอ กรุณาถ่ายใหม่"
        }

    # threshold แยกตามคลาส
    class_thresholds = {
        "under": 0.55,
        "normal": 0.45,
        "over": 0.50
    }

    threshold = class_thresholds.get(cls_name, 0.5)

    if confidence < threshold:
        return {
            "ok": False,
            "error": "low_confidence",
            "message": "ไม่สามารถประเมินได้อย่างมั่นใจ กรุณาถ่ายภาพใหม่"
        }

    return {
        "ok": True,
        "status": {
            "under": "ต่ำกว่าเกณฑ์",
            "normal": "สมส่วน",
            "over": "สูงกว่าเกณฑ์"
        }[cls_name],
        "confidence": round(float(confidence), 3)
    }
