from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "ICIAR2018_BACH_Challenge_TestDataset" / "Photos"
HEAD_PATH = ROOT / "best_saved_models" / "Macenko" / "Xception" / "xception_upto95frozen.h5"
OUTPUT_PATH = ROOT / "next_10_xception_predictions.csv"
CLASS_NAMES = ["Benign", "In-situ carcinoma", "Invasive carcinoma", "Normal tissue"]


def build_feature_extractor():
    image_input = tf.keras.Input(shape=(512, 512, 3))
    backbone = tf.keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_tensor=image_input,
    )
    pooled = [
        tf.keras.layers.GlobalAveragePooling2D()(backbone.get_layer(layer_name).output)
        for layer_name in ("block4_sepconv1_act", "block5_sepconv1_act", "block14_sepconv1")
    ]
    return tf.keras.Model(image_input, tf.keras.layers.Concatenate()(pooled))


def load_images():
    image_paths = sorted(IMAGE_DIR.glob("test*.tif"), key=lambda path: int(path.stem[4:]))
    if len(image_paths) < 20:
        raise FileNotFoundError(f"Expected at least 20 files named test0.tif through test19.tif in {IMAGE_DIR}")

    images = []
    selected_paths = image_paths[10:20]
    for image_path in selected_paths:
        image = Image.open(image_path).convert("RGB").resize((512, 512))
        images.append(np.asarray(image, dtype=np.float32) / 255.0)
    return selected_paths, np.stack(images)


def main():
    image_paths, images = load_images()
    feature_extractor = build_feature_extractor()
    classifier = tf.keras.models.load_model(HEAD_PATH, compile=False)
    features = feature_extractor.predict(images, batch_size=1, verbose=1)
    probabilities = classifier.predict(features, batch_size=1, verbose=0)
    predictions = probabilities.argmax(axis=1)

    result = pd.DataFrame(
        {
            "image": [path.name for path in image_paths],
            "predicted_class_id": predictions,
            "predicted_class": [CLASS_NAMES[index] for index in predictions],
            **{f"probability_{name}": probabilities[:, index] for index, name in enumerate(CLASS_NAMES)},
        }
    )
    result.to_csv(OUTPUT_PATH, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved predictions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()