import os

import cv2
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


IMG_SIZE = (224, 224)


def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return img_array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        raise ValueError("Gradients could not be calculated for Grad CAM.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)

    if float(max_value) == 0:
        return heatmap.numpy()

    heatmap = heatmap / max_value

    return heatmap.numpy()


def create_gradcam_overlay(image_path, heatmap, alpha=0.4):
    original_img = cv2.imread(image_path)

    if original_img is None:
        raise ValueError(f"Could not load image from path: {image_path}")

    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    heatmap_resized = cv2.resize(
        heatmap,
        (original_img.shape[1], original_img.shape[0])
    )

    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_coloured = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_coloured = cv2.cvtColor(
        heatmap_coloured,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original_img,
        1 - alpha,
        heatmap_coloured,
        alpha,
        0
    )

    return original_img, heatmap_coloured, overlay


def generate_gradcam_panel(
    image_path,
    model,
    last_conv_layer_name,
    predicted_index,
    output_folder,
    output_name
):
    os.makedirs(output_folder, exist_ok=True)

    img_array = load_and_preprocess_image(image_path)

    heatmap = make_gradcam_heatmap(
        img_array,
        model,
        last_conv_layer_name,
        predicted_index
    )

    original_img, heatmap_coloured, overlay = create_gradcam_overlay(
        image_path,
        heatmap
    )

    output_filename = f"gradcam_{output_name}.png"
    output_path = os.path.join(output_folder, output_filename)

    figure = plt.figure(figsize=(15, 5))

    try:
        plt.subplot(1, 3, 1)
        plt.imshow(original_img)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(heatmap_coloured)
        plt.title("Grad CAM Heatmap")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title("Grad CAM Overlay")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    finally:
        plt.close(figure)

    return output_filename