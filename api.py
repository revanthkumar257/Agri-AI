from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from fastapi.responses import HTMLResponse


app = FastAPI(title="Agri AI Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _find_new_model_path() -> str:
    # Prefer explicit env var
    env_path = os.environ.get("AGRI_AI_MODEL")
    if env_path and os.path.exists(env_path):
        return env_path

    # Search under D:\\agriai\\modelsu for a final *.keras file
    base = os.path.join("D:\\", "agriai", "modelsu")
    candidates: list[tuple[float, str]] = []
    if os.path.isdir(base):
        for root, _, files in os.walk(base):
            for f in files:
                if f.endswith(".keras") and f.startswith("final_"):
                    full = os.path.join(root, f)
                    try:
                        mtime = os.path.getmtime(full)
                    except OSError:
                        mtime = 0.0
                    candidates.append((mtime, full))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    # Fallback to legacy location
    return os.path.join("models", "final_model.keras")

MODEL_PATH = _find_new_model_path()
CLASSES_PATH = os.environ.get("AGRI_AI_CLASSES", os.path.join("models", "classes.txt"))

model = None
classes = []


def load_classes(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ML Model Architecture from Agri_AI (2).ipynb - Latest Implementation
def residual_block(x, filters, stride=1, conv_shortcut=False):
    """A residual block for ResNet - from Agri_AI (2).ipynb"""
    shortcut = x
    x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=stride, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if conv_shortcut:
        shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), strides=stride, padding='same', kernel_initializer='he_normal')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def build_resnet_from_scratch(input_shape, num_classes):
    """Builds a simplified ResNet-like model from scratch - from Agri_AI (2).ipynb"""
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=128, stride=2, conv_shortcut=True)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=256, stride=2, conv_shortcut=True)
    x = residual_block(x, filters=256)
    x = residual_block(x, filters=512, stride=2, conv_shortcut=True)
    x = residual_block(x, filters=512)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
    return model


@app.on_event("startup")
async def startup_event():
    global model, classes
    try:
        # Load with custom objects for the ResNet architecture
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects={
            'residual_block': residual_block,
            'build_resnet_from_scratch': build_resnet_from_scratch
        })
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}")
        # Fallback: create a new model with the correct architecture
        model = build_resnet_from_scratch(input_shape=(128, 128, 3), num_classes=38)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                     loss='categorical_crossentropy', metrics=['accuracy'])
        print("Created fallback model with ResNet architecture")
    
    classes = load_classes(CLASSES_PATH)


@app.get("/")
async def root():
    return {"ok": True, "message": "Agri AI Inference API", "num_classes": len(classes)}


@app.post("/predict")
async def predict(image: UploadFile = File(..., alias="image")):
    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        pil_image = pil_image.resize((128, 128))
        arr = np.array(pil_image).astype(np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

        probs = model.predict(arr, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        result = {
            "prediction": {
                "label": classes[pred_idx],
                "confidence": float(probs[pred_idx]),
            }
        }
        return {"ok": True, **result}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# Compatibility endpoints for existing frontend
@app.post("/api/v1/predict")
async def predict_v1(image: UploadFile = File(..., alias="image")):
    return await predict(image)

@app.get("/api/v1/classes")
async def get_classes_v1():
    return {"classes": classes, "total": len(classes)}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None, "total_classes": len(classes)}


@app.get("/ui", response_class=HTMLResponse)
async def ui_page():
    # Minimal HTML UI for direct in-browser usage
    return """
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Agri AI - Demo</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 40px; }
    .card { max-width: 640px; padding: 20px; border: 1px solid #e5e7eb; border-radius: 12px; }
    .row { margin-bottom: 16px; }
    button { padding: 10px 16px; background: #16a34a; color: white; border: 0; border-radius: 8px; cursor: pointer; }
    #result { white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace; }
    img { max-width: 100%; margin-top: 12px; border-radius: 8px; }
  </style>
  <script>
    async function doPredict(){
      const fileInput = document.getElementById('image');
      const resultEl = document.getElementById('result');
      const previewEl = document.getElementById('preview');
      resultEl.textContent = 'Predicting...';
      if(fileInput.files.length === 0){
        resultEl.textContent = 'Please choose an image.';
        return;
      }
      const file = fileInput.files[0];
      // Preview
      const reader = new FileReader();
      reader.onload = e => { previewEl.src = e.target.result; };
      reader.readAsDataURL(file);

      const form = new FormData();
      form.append('image', file, file.name);
      try{
        const res = await fetch('/predict', { method: 'POST', body: form });
        const data = await res.json();
        resultEl.textContent = JSON.stringify(data, null, 2);
      }catch(err){
        resultEl.textContent = 'Error: ' + err;
      }
    }
  </script>
  </head>
  <body>
    <div class=\"card\">
      <h2>Agri AI - Plant Disease Demo</h2>
      <div class=\"row\">
        <input id=\"image\" type=\"file\" accept=\"image/*\" />
      </div>
      <div class=\"row\">
        <button onclick=\"doPredict()\">Predict</button>
        <a href=\"/docs\" style=\"margin-left:12px\">API Docs</a>
      </div>
      <img id=\"preview\" alt=\"preview\" />
      <h3>Result</h3>
      <pre id=\"result\"></pre>
    </div>
  </body>
  </html>
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



