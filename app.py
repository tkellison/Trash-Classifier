from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

model = tf.keras.models.load_model("compost_classifier.h5")

class_names = ["Compostable", "Non-Compostable", "Recyclable"]


def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    image = preprocess_image(image)

    predictions = model.predict(image)
    label = class_names[np.argmax(predictions)]

    return {"prediction": label}
