import io
from contextlib import asynccontextmanager
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Constants
MODEL_PATH = 'mobilenetv2.keras'
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
MODEL = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    print("Loading TensorFlow model...")
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded ✅ (MobileNetV2)")
    yield
    print("Shutting down...")

app = FastAPI(title="CIFAR-10 Vision API", version="1.0", lifespan=lifespan)

# Allow all origins for local development (React Vite ports 3000/5173/5174/5175)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"success": True, "service": "vision-api", "model_loaded": MODEL is not None}

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    # 1. Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # 2. Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 3. Preprocess (Resize to 96x96, convert to array, scale)
        image = image.resize((96, 96))
        img_array = np.array(image)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        
        # 4. Predict
        predictions = MODEL.predict(img_array, verbose=0)
        predicted_idx = int(np.argmax(predictions[0]))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(np.max(predictions[0]))
        
        return {
            "success": True,
            "data": {
                "label": predicted_class,
                "confidence": round(confidence, 4)
            }
        }
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": "Failed to process image"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)