def decide(score, threshold=0.5):
    if score >= threshold:
        return {"status": "pass", "confidence": float(score)}
    else:
        return {"status": "fail", "confidence": float(score)}
