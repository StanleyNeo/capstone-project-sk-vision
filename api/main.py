# Day 12: FastAPI service for the CIFAR-10 MobileNetV2 classifier.
# Loads mobilenetv2.tflite via tflite-runtime (no TensorFlow at runtime).
import io
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tflite_runtime.interpreter import Interpreter

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / 'api' / '.env')

PORT = int(os.getenv('PORT', '5001'))
CLIENT_URLS = [u.strip() for u in os.getenv(
    'CLIENT_URLS', 'http://localhost:3000,http://localhost:5173,http://localhost:5174'
).split(',')]

MODEL_PATH = ROOT / 'mobilenetv2.tflite'
CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

MODEL = None
INPUT_DETAILS = None
OUTPUT_DETAILS = None
MODEL_META = {}


def err(status, msg):
    return JSONResponse(status_code=status, content={'success': False, 'error': msg})


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, INPUT_DETAILS, OUTPUT_DETAILS
    MODEL = Interpreter(model_path=str(MODEL_PATH))
    MODEL.allocate_tensors()
    INPUT_DETAILS = MODEL.get_input_details()
    OUTPUT_DETAILS = MODEL.get_output_details()
    MODEL_META.update({
        'model_name': 'cifar10-mobilenetv2-tflite',
        'version': '1.0',
        'algorithm': 'MobileNetV2 transfer learning (TFLite)',
        'classes': len(CLASS_NAMES),
    })

    print('=' * 56)
    print(' CIFAR-10 VISION API (PYTHON/FASTAPI)  v1.0')
    print('=' * 56)
    print(f' Port: {PORT}     URL: http://localhost:{PORT}')
    print(f' Model loaded OK  ({MODEL_META[\"algorithm\"]})')
    print(f' Input: {INPUT_DETAILS[0][\"shape\"]}  Output: {OUTPUT_DETAILS[0][\"shape\"]}')
    print(' Endpoints:')
    print('   GET  /health      service + model status')
    print('   POST /classify    multipart image -> {label, confidence}')
    print('=' * 56)
    yield


app = FastAPI(title='cifar10-vision-api', version='1.0', lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CLIENT_URLS,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.exception_handler(404)
async def not_found(request, exc):
    return err(404, f'Route not found: {request.method} {request.url.path}')


@app.get('/health')
async def health():
    return {
        'success': True,
        'service': 'cifar10-vision-api',
        'version': '1.0',
        'model_loaded': MODEL is not None,
        'model': MODEL_META,
        'time': datetime.now(timezone.utc).isoformat(),
    }


@app.post('/classify')
async def classify_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='File must be an image')

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image = image.resize((96, 96))
        img_array = np.array(image, dtype=np.float32)
        img_array = (img_array / 127.5) - 1.0
        img_array = np.expand_dims(img_array, axis=0)

        MODEL.set_tensor(INPUT_DETAILS[0]['index'], img_array)
        MODEL.invoke()
        predictions = MODEL.get_tensor(OUTPUT_DETAILS[0]['index'])[0]

        predicted_idx = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(np.max(predictions))

        return {
            'success': True,
            'data': {
                'label': predicted_class,
                'confidence': round(confidence, 4),
            },
        }
    except Exception as e:
        print(f'Error: {e}')
        return JSONResponse(status_code=500, content={'success': False, 'error': 'Failed to process image'})


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=PORT)
