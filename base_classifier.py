import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Initialize MobileNetV2
model = MobileNetV2(weights="imagenet")

# Function to classify image
def classify_image(img_array):
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return decoded_predictions

# Function to load and preprocess image
def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to apply Grad-CAM
def grad_cam(image_path, model):
    img_array = load_and_preprocess_image(image_path)

    grad_model = tf.keras.models.Model([
        model.inputs], [model.get_layer('Conv_1').output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    if np.max(heatmap) != 0:
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    else:
        heatmap = np.maximum(heatmap, 0)

    heatmap = cv2.resize(heatmap if isinstance(heatmap, np.ndarray) else heatmap.numpy(), (224, 224))
    return heatmap

# Occlusion functions
def apply_black_box(image_array, heatmap):
    masked_img = image_array.copy()
    mask = (heatmap > 0.5).astype(np.uint8)
    for c in range(3):
        masked_img[0, :, :, c] *= (1 - mask)
    return masked_img

def apply_gaussian_blur(image_array, heatmap):
    img = image_array[0].astype(np.uint8)
    mask = (heatmap > 0.5).astype(np.uint8) * 255
    blurred = cv2.GaussianBlur(img, (21, 21), 0)
    result = np.where(mask[:, :, np.newaxis] == 255, blurred, img)
    return np.expand_dims(result, axis=0)

def apply_noise(image_array, heatmap):
    img = image_array[0]
    mask = (heatmap > 0.5).astype(np.uint8)
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    noisy_img = img * (1 - mask[:, :, np.newaxis]) + noise * mask[:, :, np.newaxis]
    return np.expand_dims(noisy_img, axis=0)

# Main workflow
if __name__ == "__main__":
    image_path = "basic_cat.jpg"
    img_array = load_and_preprocess_image(image_path)

    # Apply Grad-CAM
    heatmap = grad_cam(image_path, model)

    # Apply occlusions
    black_box_img = apply_black_box(img_array, heatmap)
    gaussian_blur_img = apply_gaussian_blur(img_array, heatmap)
    noise_img = apply_noise(img_array, heatmap)

    # Classify occluded images
    occlusions = {
        "Original": img_array,
        "Black Box": black_box_img,
        "Gaussian Blur": gaussian_blur_img,
        "Noise": noise_img
    }

    # Display results with proper color mapping
for name, img in occlusions.items():
    predictions = classify_image(img)
    print(f"\n{name} Image Predictions:")
    for i, (imagenet_id, label, score) in enumerate(predictions):
        print(f"{i + 1}: {label} ({score:.2f})")

    # Ensure image is in uint8 and apply proper color conversion
    img_display = img[0]
    
    # Normalize if values exceed range
    if img_display.max() > 255:
        img_display = cv2.normalize(img_display, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert BGR to RGB for proper display in matplotlib
    img_display = cv2.cvtColor(img_display.astype(np.uint8), cv2.COLOR_BGR2RGB)

    plt.imshow(img_display)
    plt.title(f"{name} Image")
    plt.axis('off')
    plt.show()

