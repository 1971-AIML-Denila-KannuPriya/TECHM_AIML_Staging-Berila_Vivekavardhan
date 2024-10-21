from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('X-ray model.h5')  # Make sure the path is correct

def prepare_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/", methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template("index.html", prediction="No file uploaded")
        file = request.files['file']
        if file and file.filename != '':
            filepath = os.path.join("uploads", file.filename)
            file.save(filepath)

            img_array = prepare_image(filepath)
            prediction = model.predict(img_array)[0][0]

            result = "Pneumonia" if prediction > 0.5 else "Normal"
            return render_template("index.html", prediction=f"Prediction: {result}")

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
