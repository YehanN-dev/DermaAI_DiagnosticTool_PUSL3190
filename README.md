# DermaAI Diagnostic Support Tool

## Project Overview

DermaAI is an academic research prototype designed to support skin lesion analysis using deep learning and explainable AI. The system allows users to upload dermoscopic skin lesion images, receive a model based prediction, view probability scores across seven lesion categories, and inspect Grad CAM visual explanations.

The project was developed as a final year research prototype under PUSL3190 to demonstrate the integration of medical image classification, explainable AI, model evaluation, and Flask based web deployment.

This system is intended for academic demonstration and decision support exploration only. It is not designed for independent medical diagnosis.

## Key Features

1. Dermoscopic image upload through a Flask web interface.
2. Skin lesion classification using a fine tuned MobileNetV2 model.
3. Prediction confidence score and probability distribution.
4. Classification across seven HAM10000 lesion classes.
5. Grad CAM visualisation to highlight image regions influencing the prediction.
6. SHAP based secondary explainability evidence.
7. Decision support flags based on confidence and prediction risk.
8. Loading animation during image analysis.
9. Input validation for missing files and unsupported file types.
10. Medical disclaimer and research prototype warning.
11. About, Documentation, Help and SHAP Evidence pages.

## Dataset

The project uses the HAM10000 skin lesion dataset.

The seven lesion classes used in the project are:

1. `akiec`  
   Actinic keratoses and intraepithelial carcinoma

2. `bcc`  
   Basal cell carcinoma

3. `bkl`  
   Benign keratosis like lesions

4. `df`  
   Dermatofibroma

5. `mel`  
   Melanoma

6. `nv`  
   Melanocytic nevi

7. `vasc`  
   Vascular lesions

## Model

The final selected model is a fine tuned MobileNetV2 image classification model.

The model was trained and evaluated using prepared HAM10000 dataset splits. The final test accuracy recorded during experimentation was 64.23 percent.

The model file used by the Flask application is:

```text
models/mobilenetv2_finetuned_final_model.keras
```

## Explainability

Grad CAM is used as the primary real time explainability method in the Flask application. It generates a heatmap showing the image regions that contributed strongly to the model prediction. Warmer areas in the visual output indicate stronger model attention.

SHAP was implemented as a secondary explainability method during the experimental stage. It was used to provide supporting evidence by showing pixel level attribution patterns for a selected sample image. Due to its higher computational cost for image based deep learning, SHAP was not selected as the live explanation method in the Flask workflow. Instead, Grad CAM was used for the deployed prototype because it is faster and more suitable for real time visual explanation.

## Model Improvement and Calibration

The project also includes a model improvement and calibration notebook. This section evaluates prediction confidence, calibration behaviour and test time augmentation behaviour.

The final system keeps the fine tuned MobileNetV2 model as the selected classification model, while using calibration results and confidence based decision support flags to improve interpretability.

## Project Structure

```text
DermaAI_DiagnosticTool

flask_app
    app.py
    config.py

    services
        gradcam_service.py
        prediction_service.py

    static
        css
            style.css

        uploads
            uploaded image files generated during testing

        outputs
            Grad CAM output files generated during testing

        shap_outputs
            SHAP evidence outputs copied from the SHAP notebook

    templates
        index.html
        result.html
        about.html
        documentation.html
        help.html
        shap_evidence.html

models
    mobilenetv2_finetuned_final_model.keras

notebooks
    01_dataset_exploration.ipynb
    02_mobilenetv2_training.ipynb
    03_gradcam_explainability.ipynb
    04_shap_explainability.ipynb
    05_model_improvement_and_calibration.ipynb

outputs
    calibration_outputs
    gradcam_outputs
    shap_outputs
    splits

screenshots

requirements.txt

README.md
```

## How to Run the Flask Application

Open PowerShell in the project root folder:

```powershell
cd E:\DermaAI_DiagnosticTool
```

Install the required packages:

```powershell
py -3.10 -m pip install -r requirements.txt
```

Run the Flask application:

```powershell
py -3.10 flask_app\app.py
```

Open the application in a browser:

```text
http://127.0.0.1:5000
```

## How to Use the System

1. Open the upload page.
2. Select or drag a dermoscopic image into the upload area.
3. Click Analyse Image.
4. Wait while the model generates the prediction and Grad CAM explanation.
5. Review the primary prediction, confidence score and probability distribution.
6. Review the Grad CAM visualisation.
7. Review the model interpretation section and decision support flags.
8. Use the About, Documentation, Help and SHAP Evidence pages for supporting information.

## Supported File Types

The system supports:

1. JPG
2. JPEG
3. PNG

Maximum upload size:

```text
8 MB
```

## Application Pages

The Flask application includes the following pages:

1. Upload page  
   Allows the user to upload a dermoscopic image for analysis.

2. Result page  
   Displays the uploaded lesion image, prediction result, confidence score, probability distribution, Grad CAM visualisation and decision support flags.

3. About page  
   Explains the purpose, model, explainability approach and limitations of the system.

4. Documentation page  
   Summarises the system workflow from image upload to Grad CAM explanation.

5. Help page  
   Provides guidance on supported file types, image analysis and confidence interpretation.

6. SHAP Evidence page  
   Presents SHAP as a secondary explainability method used during experimentation.

## Validation Handling

The system includes validation for common upload issues.

1. If no image is selected, the application displays an error message asking the user to select an image before analysis.
2. If an unsupported file type is uploaded, the application displays an error message requesting a JPG, JPEG or PNG image.
3. If the uploaded image exceeds the size limit, the application displays an error message asking for an image below 8 MB.
4. If an image cannot be analysed, the application displays an error message requesting another valid dermoscopic image.

## Final Test Evidence

The final system evidence includes screenshots of:

1. Upload page
2. File selected state
3. Loading animation
4. Result page top section
5. Grad CAM and model interpretation section
6. About page
7. Documentation page
8. Help page
9. No image selected validation error
10. Unsupported file type validation error
11. SHAP Evidence page

## Medical Disclaimer

This system provides AI assisted decision support only. It does not replace dermatologist assessment, biopsy, clinical history or professional diagnosis. Clinical confirmation is required before any medical decision.

## Limitations

1. The system is a research prototype and is not intended for clinical deployment.
2. The model accuracy is limited by the dataset, class imbalance and training configuration.
3. Predictions should be reviewed by a qualified medical professional.
4. Grad CAM provides visual explanation support, but it does not prove clinical correctness.
5. SHAP was used as supporting explainability evidence and not as the real time deployed explanation method.
6. The system currently analyses uploaded images only and does not include patient history or clinical metadata.
7. The application is designed for local academic demonstration and has not been validated for production deployment.

## Final Notes

DermaAI demonstrates the integration of deep learning, Flask based web deployment, medical image classification, probability based decision support and explainable AI visualisation within a single academic prototype.

The system shows how a trained MobileNetV2 model can be connected to a usable diagnostic support interface while maintaining transparency through Grad CAM and SHAP based explainability evidence.