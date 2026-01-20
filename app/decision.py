def decide(
    cls_name,
    confidence,
    face_ok,
    quality_ok,
    probs=None,
    debug=None
):
    """
    cls_name   : str | None   -> 'under' | 'normal' | 'over'
    confidence : float        -> max probability (0-1)
    face_ok    : bool         -> ตรวจเจอหน้าไหม
    quality_ok : bool         -> ภาพผ่านคุณภาพไหม
    probs      : list | dict  -> raw probability จากโมเดล
    debug      : dict         -> debug จากขั้นก่อนหน้า
    """

    if debug is None:
        debug = {}

    # =========================
    # 🔍 IMAGE GATE DEBUG
    # =========================
    debug["image_check"] = {
        "face_detected": face_ok,
        "quality_ok": quality_ok
    }

    # ❌ ไม่พบหน้า
    if not face_ok:
        debug["decision"] = {"final": "reject_no_face"}
        return {
            "ok": False,
            "reason": "no_face",
            "message": "ไม่พบใบหน้าในภาพงับ ลองถ่ายใหม่ให้เห็นหน้าชัด ๆ นะคะ 💛",
            "debug": debug
        }

    # ❌ คุณภาพภาพไม่ผ่าน
    if not quality_ok:
        debug["decision"] = {"final": "reject_bad_quality"}
        return {
            "ok": False,
            "reason": "bad_quality",
            "message": "ภาพไม่ชัดพอเลยงับ ลองถ่ายในที่สว่างขึ้นนิดนึงนะคะ ✨",
            "debug": debug
        }

    # =========================
    # 🧠 MODEL OUTPUT DEBUG
    # =========================
    debug["model_output"] = {
        "predicted_class": cls_name,
        "confidence": round(float(confidence), 4),
        "probs": probs
    }

    # ❌ โมเดลไม่มั่นใจ / ไม่กล้าทาย
    if cls_name is None:
        debug["decision"] = {"final": "reject_low_confidence"}
        return {
            "ok": False,
            "reason": "low_confidence",
            "confidence": round(confidence * 100, 2),
            "message": "พี่ป๊อปยังไม่มั่นใจพอจากภาพนี้งับ ลองถ่ายใหม่อีกครั้งนะคะ 💛",
            "debug": debug
        }

    # =========================
    # ⚖️ RULE-BASED ADJUSTMENT
    # =========================
    adjusted_class = cls_name
    rule_applied = "direct_prediction"

    # ถ้าคะแนนสูสีกันมาก (เช่น under vs normal)
    if probs:
        try:
            if isinstance(probs, dict):
                u = probs.get("under", 0)
                n = probs.get("normal", 0)
                o = probs.get("over", 0)
            else:
                u, n, o = probs

            # under / normal สูสี → ดันเป็น normal
            if abs(u - n) < 0.1 and max(u, n) > o:
                adjusted_class = "normal"
                rule_applied = "ambiguous_under_normal"
        except Exception:
            rule_applied = "prob_parse_error"

    # =========================
    # ✅ FINAL DECISION
    # =========================
    LABEL_TEXT = {
        "under": "ต่ำกว่าเกณฑ์",
        "normal": "สมส่วน",
        "over": "สูงกว่าเกณฑ์"
    }

    MESSAGE_TEXT = {
        "under": "ไม่ต้องกังวลนะงับ ร่างกายแต่ละคนไม่เหมือนกัน ค่อย ๆ ดูแลไปทีละนิด 💛",
        "normal": "ดีมากเลยงับ ดูแลตัวเองได้ดีแล้ว รักษาความสมดุลแบบนี้ไว้นะคะ ✨",
        "over": "ไม่เป็นไรเลยงับ สุขภาพค่อย ๆ ปรับได้ ทีละก้าวก็พอ 💛"
    }

    debug["decision"] = {
        "final_class": adjusted_class,
        "rule_applied": rule_applied
    }

    return {
        "ok": True,
        "status": LABEL_TEXT.get(adjusted_class, adjusted_class),
        "confidence": round(confidence * 100, 2),
        "message": MESSAGE_TEXT.get(adjusted_class, ""),
        "debug": debug
    }
