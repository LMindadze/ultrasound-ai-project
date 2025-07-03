import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from utils.preprocessing import extract_roi

IMG_SIZE = (224, 224)

def preprocess_image(path):
    img = extract_roi(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return img.astype(np.float32) / 255.0

def main():
    root = os.path.dirname(os.path.dirname(__file__))
    val_dir = os.path.join(root, "data_validation")

    # Load models
    organ_model  = load_model(os.path.join(root, "models", "organ_classifier.h5"))
    kidney_model = load_model(os.path.join(root, "models", "kidney_diagnosis.h5"))
    liver_model  = load_model(os.path.join(root, "models", "liver_diagnosis.h5"))

    diag_classes = {
        "kidney": ["normal", "stone"],
        "liver":  ["normal", "benign", "malignant"]
    }

    entries = []
    for organ in ("kidney", "liver"):
        organ_dir = os.path.join(val_dir, organ)
        if not os.path.isdir(organ_dir): continue
        for sub in os.listdir(organ_dir):
            sub_dir = os.path.join(organ_dir, sub)
            if not os.path.isdir(sub_dir): continue
            for fname in os.listdir(sub_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    entries.append((os.path.join(sub_dir, fname), organ, sub, fname))

    total = len(entries)
    organ_corr = diag_corr = both_corr = 0
    kidney_total = kidney_correct = 0
    liver_total = liver_correct = 0

    print("\n📊 Full Pipeline Results\n")
    for path, true_organ, true_diag, fname in entries:
        img = preprocess_image(path)[None]  # add batch dim

        # Stage 1
        pred_o = ("kidney", "liver")[np.argmax(organ_model.predict(img, verbose=0))]
        ok_o = (pred_o == true_organ)
        if ok_o: organ_corr += 1

        # Stage 2
        if pred_o == "kidney":
            preds = kidney_model.predict(img, verbose=0)
        else:
            preds = liver_model.predict(img, verbose=0)

        pred_diag = diag_classes[pred_o][np.argmax(preds)]
        ok_d = (pred_diag == true_diag)
        if ok_d: diag_corr += 1

        if pred_o == "kidney":
            kidney_total += 1
            if ok_d: kidney_correct += 1
        elif pred_o == "liver":
            liver_total += 1
            if ok_d: liver_correct += 1

        ok_both = ok_o and ok_d
        if ok_both: both_corr += 1

        print(
            f"{true_organ} -> {true_diag} -> {fname} -> "
            f"OrganPred: {pred_o} ({'OK' if ok_o else 'FAIL'}) ; "
            f"DiagPred: {pred_diag} ({'OK' if ok_d else 'FAIL'}) -> "
            f"{'✅ Passed' if ok_both else '❌ Failed'}"
        )

    # Summary
    print("\n🔍 Summary:")
    print(f"  Total images:            {total}")
    print(f"  Organ stage correct:     {organ_corr}/{total} ({organ_corr/total*100:.2f}%)")


    if kidney_total > 0:
        print(f"  Kidney model accuracy:   {kidney_correct}/{kidney_total} ({kidney_correct/kidney_total*100:.2f}%)")
    if liver_total > 0:
        print(f"  Liver model accuracy:    {liver_correct}/{liver_total} ({liver_correct/liver_total*100:.2f}%)")

    print(f"  Diagnosis stage correct: {diag_corr}/{total} ({diag_corr/total*100:.2f}%)")
    print(f"  Both correct:            {both_corr}/{total} ({both_corr/total*100:.2f}%)")

if __name__ == "__main__":
    main()
