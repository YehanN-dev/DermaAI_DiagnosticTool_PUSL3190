import uuid
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from config import UPLOAD_FOLDER, OUTPUT_FOLDER, ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH
from services.prediction_service import model, predict_image, get_last_conv_layer_name
from services.gradcam_service import generate_gradcam_panel


app = Flask(__name__)

app.secret_key = "dermaai_local_demo_secret_key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

LAST_CONV_LAYER_NAME = get_last_conv_layer_name()


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    flash("The uploaded image is too large. Please upload an image below 8 MB.")
    return redirect(url_for("index"))


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        flash("No image file was uploaded. Please select an image before submitting.")
        return redirect(url_for("index"))

    uploaded_file = request.files["image"]

    if uploaded_file.filename == "":
        flash("No image file was uploaded. Please select an image before submitting.")
        return redirect(url_for("index"))

    if not allowed_file(uploaded_file.filename):
        flash("Unsupported file type. Please upload a JPG, JPEG or PNG image.")
        return redirect(url_for("index"))

    original_filename = secure_filename(uploaded_file.filename)

    if "." not in original_filename:
        flash("Unsupported file type. Please upload a JPG, JPEG or PNG image.")
        return redirect(url_for("index"))

    file_extension = original_filename.rsplit(".", 1)[1].lower()

    unique_id = uuid.uuid4().hex
    saved_filename = f"{unique_id}.{file_extension}"
    saved_path = UPLOAD_FOLDER / saved_filename

    try:
        uploaded_file.save(saved_path)

        prediction_result = predict_image(str(saved_path))

        gradcam_filename = None

        if not prediction_result["input_warning"]:
            gradcam_filename = generate_gradcam_panel(
                image_path=str(saved_path),
                model=model,
                last_conv_layer_name=LAST_CONV_LAYER_NAME,
                predicted_index=prediction_result["predicted_index"],
                output_folder=str(OUTPUT_FOLDER),
                output_name=unique_id
            )

    except Exception as error:
        print("Prediction error:", error)
        flash("The image could not be analysed. Please try another valid dermoscopic image.")
        return redirect(url_for("index"))

    uploaded_image_url = url_for("static", filename=f"uploads/{saved_filename}")

    gradcam_image_url = None

    if gradcam_filename:
        gradcam_image_url = url_for("static", filename=f"outputs/{gradcam_filename}")

    return render_template(
        "result.html",
        result=prediction_result,
        uploaded_image_url=uploaded_image_url,
        gradcam_image_url=gradcam_image_url
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/documentation")
def documentation():
    return render_template("documentation.html")


@app.route("/shap-evidence")
def shap_evidence():
    shap_static_folder = Path(app.root_path) / "static" / "shap_outputs"
    shap_static_folder.mkdir(parents=True, exist_ok=True)

    shap_images = sorted(shap_static_folder.glob("*.png"))
    shap_tables = sorted(shap_static_folder.glob("*.csv"))

    shap_image_url = None
    shap_table_url = None

    if shap_images:
        shap_image_url = url_for(
            "static",
            filename=f"shap_outputs/{shap_images[0].name}"
        )

    if shap_tables:
        shap_table_url = url_for(
            "static",
            filename=f"shap_outputs/{shap_tables[0].name}"
        )

    return render_template(
        "shap_evidence.html",
        shap_image_url=shap_image_url,
        shap_table_url=shap_table_url
    )


@app.route("/help")
def help_page():
    return render_template("help.html")


if __name__ == "__main__":
    app.run(debug=True)