import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from config import MODEL_PATH

IMG_SIZE = (224, 224)

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

LESION_TYPE_DICT = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions"
}

HIGH_RISK_CLASSES = ["mel", "bcc", "akiec"]

model = load_model(MODEL_PATH)


def load_single_image(image_path):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def generate_decision_support_flags(probability_table):
    top_1 = probability_table.iloc[0]
    top_2 = probability_table.iloc[1]

    predicted_label = top_1["class_code"]
    confidence = float(top_1["probability"])
    second_confidence = float(top_2["probability"])
    margin = confidence - second_confidence

    flags = []

    if confidence < 0.50:
        flags.append("Low confidence prediction. Clinical review is strongly recommended.")

    if margin < 0.15:
        flags.append("Uncertainty detected between the top two predicted classes.")

    if predicted_label in HIGH_RISK_CLASSES:
        flags.append("Potentially higher risk class predicted. Urgent clinical review is recommended.")

    if len(flags) == 0:
        flags.append("Prediction confidence is acceptable, but clinical review is still required.")

    return {
        "second_label": top_2["class_code"],
        "second_confidence_percent": round(second_confidence * 100, 2),
        "margin_percent": round(margin * 100, 2),
        "flags": flags
    }


def get_last_conv_layer_name():
    conv_layers = [
        layer.name for layer in model.layers
        if isinstance(layer, tf.keras.layers.Conv2D)
    ]

    return conv_layers[-1]


def predict_image(image_path):
    img_array = load_single_image(image_path)

    probabilities = model.predict(img_array, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    probability_table = pd.DataFrame({
        "class_code": CLASS_NAMES,
        "class_name": [LESION_TYPE_DICT[class_code] for class_code in CLASS_NAMES],
        "probability": probabilities
    })

    probability_table["probability_percent"] = (
        probability_table["probability"] * 100
    ).round(2)

    probability_table = probability_table.sort_values(
        by="probability",
        ascending=False
    ).reset_index(drop=True)

    support = generate_decision_support_flags(probability_table)

    return {
        "predicted_index": predicted_index,
        "predicted_label": predicted_label,
        "predicted_class_name": LESION_TYPE_DICT[predicted_label],
        "confidence_percent": round(confidence * 100, 2),
        "probability_table": probability_table.to_dict(orient="records"),
        "top_3": probability_table.head(3).to_dict(orient="records"),
        "flags": support["flags"],
        "second_label": support["second_label"],
        "second_confidence_percent": support["second_confidence_percent"],
        "margin_percent": support["margin_percent"]
    }
