import torch
import torch.nn.functional as F

# class index ต้องตรงตอน train
CLASS_NAMES = ["bad", "ok", "good"]

# confidence ขั้นต่ำ
CONF_THRESHOLD = 0.65

def decide(logits):
    probs = F.softmax(logits, dim=1)
    conf, pred = torch.max(probs, dim=1)

    confidence = conf.item()
    label_idx = pred.item()
    label = CLASS_NAMES[label_idx]

    if confidence < CONF_THRESHOLD:
        return {
            "result": "reject",
            "reason": "ความมั่นใจต่ำ",
            "confidence": confidence
        }

    return {
        "result": "accept",
        "label": label,
        "confidence": confidence
    }
