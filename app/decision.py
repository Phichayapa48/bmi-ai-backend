def decide(cls_name, confidence, threshold=0.4):
    """
    cls_name: 'under' | 'normal' | 'over'
    confidence: float (0-1)
    """

    if confidence < threshold:
        return {
            "ok": False,
            "message": "ไม่มั่นใจในผลลัพธ์ กรุณาถ่ายภาพใหม่"
        }

    return {
        "ok": True,
        "class": cls_name,
        "confidence": round(float(confidence), 3)
    }
