import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from utils.preprocessing import extract_roi

def run_full_pipeline(val_dir, model_paths):
    IMG_SIZE = (224, 224)
    organ_model = load_model(model_paths["organ"])
    kidney_model = load_model(model_paths["kidney"])
    liver_model = load_model(model_paths["liver"])

    diag_classes = {
        "kidney": ["normal", "stone"],
        "liver":  ["normal", "benign", "malignant"]
    }

    entries = []
    for organ in ("kidney", "liver"):
        organ_dir = os.path.join(val_dir, organ)
        for sub in os.listdir(organ_dir):
            sub_dir = os.path.join(organ_dir, sub)
            for fname in os.listdir(sub_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    entries.append((os.path.join(sub_dir, fname), organ, sub, fname))

    results = []
    for path, true_organ, true_diag, fname in entries:
        img = extract_roi(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE).astype(np.float32) / 255.0
        img_batch = img[None]

        pred_organ = ("kidney", "liver")[np.argmax(organ_model.predict(img_batch, verbose=0))]
        pred_diag = diag_classes[pred_organ][np.argmax(
            kidney_model.predict(img_batch, verbose=0) if pred_organ == "kidney"
            else liver_model.predict(img_batch, verbose=0)
        )]

        results.append({
            "True Organ": true_organ,
            "True Diagnosis": true_diag,
            "File": fname,
            "Predicted Organ": pred_organ,
            "Organ OK": "✅" if pred_organ == true_organ else "❌",
            "Predicted Diagnosis": pred_diag,
            "Diagnosis OK": "✅" if pred_diag == true_diag else "❌"
        })

    df = pd.DataFrame(results)

    # Compute statistics
    total = len(df)
    organ_corr = df['Organ OK'].value_counts().get("✅", 0)
    diag_corr = df['Diagnosis OK'].value_counts().get("✅", 0)
    both_corr = ((df["Organ OK"] == "✅") & (df["Diagnosis OK"] == "✅")).sum()

    kidney_df = df[df["True Organ"] == "kidney"]
    liver_df = df[df["True Organ"] == "liver"]
    kidney_correct = (kidney_df["Diagnosis OK"] == "✅").sum()
    liver_correct = (liver_df["Diagnosis OK"] == "✅").sum()

    summary = {
        "Total images": total,
        "Organ stage correct": f"{organ_corr}/{total} ({organ_corr/total*100:.2f}%)",
        "Kidney model accuracy": f"{kidney_correct}/{len(kidney_df)} ({kidney_correct/len(kidney_df)*100:.2f}%)",
        "Liver model accuracy": f"{liver_correct}/{len(liver_df)} ({liver_correct/len(liver_df)*100:.2f}%)",
        "Diagnosis stage correct": f"{diag_corr}/{total} ({diag_corr/total*100:.2f}%)",
        "Both correct": f"{both_corr}/{total} ({both_corr/total*100:.2f}%)"
    }

    return df, summary
