from urllib import response
from flask import Flask, jsonify, render_template, request
import pickle
import base64
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST", "GET"])
def predict():
    with open("handwritten-digit-classifier.pkl", "rb") as f:
        model = pickle.load(f)
        image = request.get_json(silent=True)["image"].split(",")[1]
        image_data = base64.urlsafe_b64decode(image)
        image_array = Image.open(io.BytesIO(image_data))
        image_array = image_array.convert("L")
        image_array = image_array.resize((28, 28))
        image_array = np.array(image_array, dtype=np.float32).flatten()
        image_array = image_array / 255
        response = {
            "prediction": str(model.predict([image_array])[0])
        }
        return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)