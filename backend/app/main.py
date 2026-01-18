from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import traceback
import torch   # ✅ เพิ่มบรรทัดนี้

from app.model import get_model
from app.utils import preprocess_image

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok", "service": "BMI AI Backend"}

@app.on_event("startup")
def startup_event():
    print("🚀 Loading model...")
    get_model()
    print("✅ Model ready")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        x = preprocess_image(image)

        model = get_model()
        with torch.no_grad():
            y = model(x)
            bmi = float(y.squeeze().item())

        return {"bmi": bmi}

    except Exception as e:
        print("❌ Predict error")
        traceback.print_exc()
        return {
            "error": "prediction_failed",
            "message": str(e)
        }
