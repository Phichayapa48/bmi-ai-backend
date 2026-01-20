from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch

from app.model import get_model
from app.face_utils import detect_and_crop_face
from app.utils import preprocess_image
from app.quality_check import quality_check
from app.decision import decide

app = FastAPI()
model = get_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # 1️⃣ Detect face
    face = detect_and_crop_face(image)
    if face is None:
        return {
            "result": "reject",
            "reason": "ไม่พบใบหน้า"
        }

    # 2️⃣ Quality check
    ok, msg = quality_check(face)
    if not ok:
        return {
            "result": "reject",
            "reason": msg
        }

    # 3️⃣ Preprocess
    x = preprocess_image(face).to(next(model.parameters()).device)

    # 4️⃣ Predict
    with torch.no_grad():
        logits = model(x)

    # 5️⃣ Decision
    return decide(logits)
