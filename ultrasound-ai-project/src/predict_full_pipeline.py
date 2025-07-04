import sys
import os
import numpy as np
import tensorflow as tf
import cv2
import traceback

from keras.models import load_model
from utils.preprocessing import extract_roi

IMG_SIZE = (224, 224)

def preprocess_image(path):
    img = extract_roi(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return img.astype(np.float32) / 255.0

def main():
    try:
        if len(sys.argv) < 2:
            print(" No image path provided.")
            sys.exit(2)

        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f" Image not found: {image_path}")
            sys.exit(2)

        root = os.path.dirname(os.path.dirname(__file__))

        organ_model = load_model(os.path.join(root, "models", "organ_classifier.h5"))
        kidney_model = load_model(os.path.join(root, "models", "kidney_diagnosis.h5"))
        liver_model = load_model(os.path.join(root, "models", "liver_diagnosis.h5"))

        diag_classes = {
            "kidney": ["normal", "stone"],
            "liver":  ["normal", "benign", "malignant"]
        }

        img = preprocess_image(image_path)[None]  # Add batch dimension

        pred_organ = ("kidney", "liver")[np.argmax(organ_model.predict(img, verbose=0))]
        if pred_organ == "kidney":
            diag_pred = kidney_model.predict(img, verbose=0)
        else:
            diag_pred = liver_model.predict(img, verbose=0)

        pred_diag = diag_classes[pred_organ][np.argmax(diag_pred)]

        print(" Image:", os.path.basename(image_path))
        print(" Predicted Organ:", pred_organ)
        print(" Diagnosis:", pred_diag)

    except Exception as e:
        print("\n Exception occurred:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
