import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import keras

# ================== CONFIG ==================
IMG_SIZE = (224, 224)
THRESHOLD = 0.4
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pneumonia_model.keras")

# ✅ CREATE APP ONCE
app = Flask(__name__)

# ✅ APPLY CORS TO THE SAME APP
CORS(app, resources={r"/*": {"origins": "*"}})

# Load model
model = keras.models.load_model(
    MODEL_PATH,
    compile=False
)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    image_bytes = request.files["file"].read()
    img = preprocess_image(image_bytes)

    prob = float(model(img, training=False).numpy()[0][0])
    prediction = "PNEUMONIA" if prob > THRESHOLD else "NORMAL"

    return jsonify({
        "prediction": prediction,
        "probability": round(prob, 4),
        "threshold": THRESHOLD
    })

if __name__ == "__main__":
    app.run(debug=True)
