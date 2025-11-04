import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING logs

from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
from PIL import ImageOps
import io
import re
import base64

app = Flask(__name__)
model = tf.keras.models.load_model("mnist_model.keras")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if 'image' not in data:
        return jsonify({"error": "No image provided"}), 400

    image_data = re.sub('^data:image/.+;base64,', '', data['image'])
    image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('L')

    image = ImageOps.invert(image)
    image = ImageOps.fit(image, (20, 20), centering=(0.5, 0.5))
    new_image = Image.new("L", (28, 28), 0)
    new_image.paste(image, (4, 4))
    image_array = np.array(new_image)
    image_array = image_array / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    prediction = model.predict(image_array)
    digit = int(np.argmax(prediction))

    return jsonify({"prediction": digit})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)