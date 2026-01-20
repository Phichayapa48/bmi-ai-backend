def decide(cls_name, confidence, face_ok, quality_ok, debug=None):
    if not face_ok:
        return {
            "ok": False,
            "reason": "no_face",
            "message": "ไม่พบใบหน้า",
            "debug": debug
        }

    if not quality_ok:
        return {
            "ok": False,
            "reason": "low_quality",
            "message": "ภาพไม่ชัด",
            "debug": debug
        }

    if cls_name is None:
        return {
            "ok": False,
            "reason": "low_confidence",
            "message": "โมเดลไม่มั่นใจ",
            "confidence": confidence,
            "debug": debug
        }

    return {
        "ok": True,
        "result": cls_name,
        "confidence": confidence,
        "debug": debug
    }
