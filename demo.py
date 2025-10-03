import argparse
import os
import numpy as np
from PIL import Image
import tensorflow as tf


def load_classes(classes_path: str):
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"classes.txt not found at {classes_path}. Run training first.")
    with open(classes_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes


def preprocess_image(image_path: str, image_size=(128, 128)):
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(image_size)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def main():
    parser = argparse.ArgumentParser(description="Agri AI Demo")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model-path", default="D:/agriai/modelsu/AGRI AI-20250908T152325Z-1-001/AGRI AI/final_agri_ai_scratch_v2_model.keras", help="Path to trained model")
    parser.add_argument("--classes-path", default="models/classes.txt", help="Path to classes.txt")
    parser.add_argument("--image-size", type=int, nargs=2, default=[128, 128], help="Image size H W")
    args = parser.parse_args()

    classes = load_classes(args.classes_path)
    
    # Load model with custom objects for ResNet architecture
    try:
        model = tf.keras.models.load_model(args.model_path, custom_objects={
            'residual_block': None,  # Will be defined below
            'build_resnet_from_scratch': None
        })
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying to load without custom objects...")
        model = tf.keras.models.load_model(args.model_path)

    image_array = preprocess_image(args.image, image_size=tuple(args.image_size))
    probs = model.predict(image_array, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_label = classes[pred_idx]
    pred_conf = float(probs[pred_idx])

    print(f"Prediction: {pred_label} (confidence {pred_conf:.4f})")


if __name__ == "__main__":
    main()


