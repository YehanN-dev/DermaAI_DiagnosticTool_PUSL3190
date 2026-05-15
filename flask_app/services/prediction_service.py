import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from config import (
    MODEL_PATH,
    LOW_CONFIDENCE_THRESHOLD,
    VERY_LOW_CONFIDENCE_THRESHOLD,
    LOW_MARGIN_THRESHOLD
)


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


def generate_input_suitability_warning(probability_table):
    top_1 = probability_table.iloc[0]
    top_2 = probability_table.iloc[1]

    confidence = float(top_1["probability"])
    second_confidence = float(top_2["probability"])
    margin = confidence - second_confidence

    very_low_confidence = confidence < VERY_LOW_CONFIDENCE_THRESHOLD
    low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD
    uncertain_prediction = margin < LOW_MARGIN_THRESHOLD

    input_warning = very_low_confidence or (low_confidence and uncertain_prediction)

    warning_title = None
    warning_message = None
    warning_reasons = []

    if input_warning:
        warning_title = "Analysis Not Displayed"
        warning_message = (
            "The uploaded image could not be interpreted reliably by this prototype. "
            "This may be due to low model confidence, image quality, lesion ambiguity, "
            "or the image being outside the intended dermoscopic input domain. "
            "No classification has been displayed to avoid presenting a misleading result."
        )

        if very_low_confidence:
            warning_reasons.append(
                "The model confidence was very low, so the classification output was suppressed."
            )

        if low_confidence and uncertain_prediction:
            warning_reasons.append(
                "The model showed both low confidence and uncertainty between the top predicted classes."
            )

    print("=== INPUT SUITABILITY DEBUG ===")
    print("Top confidence:", confidence)
    print("Second confidence:", second_confidence)
    print("Margin:", margin)
    print("Very low confidence:", very_low_confidence)
    print("Low confidence:", low_confidence)
    print("Uncertain prediction:", uncertain_prediction)
    print("Input warning:", input_warning)
    print("===============================")

    return {
        "input_warning": bool(input_warning),
        "warning_title": warning_title,
        "warning_message": warning_message,
        "warning_reasons": warning_reasons,
        "very_low_confidence": bool(very_low_confidence),
        "low_confidence": bool(low_confidence),
        "uncertain_prediction": bool(uncertain_prediction)
    }


def get_last_conv_layer_name():
    conv_layers = [
        layer.name for layer in model.layers
        if isinstance(layer, tf.keras.layers.Conv2D)
    ]

    if not conv_layers:
        raise ValueError("No Conv2D layer was found in the model.")

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
    suitability = generate_input_suitability_warning(probability_table)

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
        "margin_percent": support["margin_percent"],
        "input_warning": suitability["input_warning"],
        "warning_title": suitability["warning_title"],
        "warning_message": suitability["warning_message"],
        "warning_reasons": suitability["warning_reasons"],
        "very_low_confidence": suitability["very_low_confidence"],
        "low_confidence": suitability["low_confidence"],
        "uncertain_prediction": suitability["uncertain_prediction"]
    }