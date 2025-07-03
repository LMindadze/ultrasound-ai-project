import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from utils.preprocessing import extract_roi

def main():
    project_root = os.path.dirname(os.path.dirname(__file__))
    validation_data_dir = os.path.join(project_root, "data_validation")
    model_path = os.path.join(project_root, "models/organ_classifier.h5")

    # Load model
    model = load_model(model_path)

    # Prepare class index mapping
    class_names = []
    image_entries = []

    for organ in os.listdir(validation_data_dir):
        for cls in os.listdir(os.path.join(validation_data_dir, organ)):
            label = f"{organ}_{cls}"
            class_names.append(label)
            folder = os.path.join(validation_data_dir, organ, cls)
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_entries.append((os.path.join(folder, fname), label, fname))

    class_names = sorted(set(class_names))
    class_to_index = {name: i for i, name in enumerate(class_names)}

    # Print header
    print("📊 Prediction Results:\n")

    correct = 0

    for path, true_label, fname in image_entries:
        try:
            # Preprocess
            img = extract_roi(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            # Predict
            preds = model.predict(img, verbose=0)
            pred_idx = np.argmax(preds)
            pred_label = class_names[pred_idx]

            # Compare
            passed = pred_label == true_label
            if passed:
                correct += 1

            print(f"{true_label.split('_')[0]} -> {true_label.split('_')[1]} -> {fname} -> {'✅ Passed' if passed else '❌ Failed (Predicted: ' + pred_label + ')'}")

        except Exception as e:
            print(f"{true_label} -> {fname} -> ❌ Error: {e}")

    # Summary
    total = len(image_entries)
    acc = (correct / total) * 100 if total > 0 else 0
    print(f"\n✅ Overall Accuracy on unseen data: {correct}/{total} ({acc:.2f}%)")

if __name__ == "__main__":
    main()
