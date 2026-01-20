def decide(cls_name, confidence, face_ok, quality_ok):
    if not face_ok:
        return {
            "ok": False,
            "reason": "no_face",
            "message": "ไม่พบใบหน้าในภาพ"
        }

    if not quality_ok:
        return {
            "ok": False,
            "reason": "bad_quality",
            "message": "คุณภาพภาพไม่เหมาะสม"
        }

    return {
        "ok": True,
        "class": cls_name,
        "confidence": round(confidence, 4)
    }
