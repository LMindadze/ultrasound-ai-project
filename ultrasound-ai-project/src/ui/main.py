import streamlit as st
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import subprocess
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report
from keras.models import load_model

from utils.preprocessing import extract_roi
from utils.dataloader import load_organ_data, load_kidney_data, load_liver_data
from ui_utils import run_full_pipeline

# --- Configuration ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TensorFlow logs
PYTHON = sys.executable
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_DIR = os.path.join(PROJ_ROOT, 'src')
DATA_VAL = os.path.join(PROJ_ROOT, 'data_validation')

st.set_page_config(page_title="Ultrasound AI Pipeline", layout="centered")
st.title("🧠 Ultrasound AI Project Interface")

# Tabs
tab1, tab2, tab3 = st.tabs(["Train","Evaluate","Predict"])

# --- Tab 1: Training ---
with tab1:
    st.header("📚 Train Models")
    if st.button("Train Organ Classifier"):
        subprocess.run([PYTHON, os.path.join(SRC_DIR, 'train.py')])
        st.success("Organ classifier trained.")
    if st.button("Train Kidney Diagnosis"):
        subprocess.run([PYTHON, os.path.join(SRC_DIR, 'train_kidney_diagnosis.py')])
        st.success("Kidney diagnosis model trained.")
    if st.button("Train Liver Diagnosis"):
        subprocess.run([PYTHON, os.path.join(SRC_DIR, 'train_liver_diagnosis.py')])
        st.success("Liver diagnosis model trained.")

# --- Helper function to pretty-print classification report ---
def render_classification_report(y_true, y_pred, target_names):
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df = df.round(2)
    st.dataframe(df.style.background_gradient(cmap="Blues"), use_container_width=True)

# --- Tab 2: Evaluation ---
with tab2:
    st.header("✅ Evaluate Models on Unseen Data")

    # Organ Classifier
    st.subheader("Organ Classifier (kidney vs liver)")
    if st.button("Evaluate Organ Classifier"):
        model = load_model(os.path.join(PROJ_ROOT, 'models', 'organ_classifier.h5'))
        X, y_true = [], []
        for organ in ['kidney','liver']:
            folder = os.path.join(DATA_VAL, organ)
            for sub in os.listdir(folder):
                sub_dir = os.path.join(folder, sub)
                for fname in os.listdir(sub_dir):
                    path = os.path.join(sub_dir, fname)
                    img = extract_roi(path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224,224)).astype(np.float32)/255.0
                    X.append(img)
                    y_true.append(0 if organ=='kidney' else 1)
        X = np.stack(X)
        y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
        acc = accuracy_score(y_true, y_pred)
        st.write(f"**Accuracy:** {acc*100:.2f}%")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm, display_labels=['kidney','liver']).plot(ax=ax)
        st.pyplot(fig)
        render_classification_report(y_true, y_pred, target_names=['kidney','liver'])

    # Kidney Model
    st.subheader("Kidney Diagnosis (normal vs stone)")
    if st.button("Evaluate Kidney Model"):
        model = load_model(os.path.join(PROJ_ROOT, 'models', 'kidney_diagnosis.h5'))
        paths, labels = [], []
        for sub in ['normal','stone']:
            folder = os.path.join(DATA_VAL, 'kidney', sub)
            for fname in os.listdir(folder):
                path = os.path.join(folder, fname)
                img = extract_roi(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224,224)).astype(np.float32)/255.0
                paths.append(img)
                labels.append(0 if sub=='normal' else 1)
        X = np.stack(paths)
        y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
        acc = accuracy_score(labels, y_pred)
        st.write(f"**Accuracy:** {acc*100:.2f}%")
        cm = confusion_matrix(labels, y_pred)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm, display_labels=['normal','stone']).plot(ax=ax)
        st.pyplot(fig)
        render_classification_report(labels, y_pred, target_names=['normal','stone'])

    # Liver Model
    st.subheader("Liver Diagnosis (normal, benign, malignant)")
    if st.button("Evaluate Liver Model"):
        model = load_model(os.path.join(PROJ_ROOT, 'models', 'liver_diagnosis.h5'))
        paths, labels = [], []
        classes = ['normal','benign','malignant']
        for idx, sub in enumerate(classes):
            folder = os.path.join(DATA_VAL, 'liver', sub)
            for fname in os.listdir(folder):
                path = os.path.join(folder, fname)
                img = extract_roi(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224,224)).astype(np.float32)/255.0
                paths.append(img)
                labels.append(idx)
        X = np.stack(paths)
        y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
        acc = accuracy_score(labels, y_pred)
        st.write(f"**Accuracy:** {acc*100:.2f}%")
        cm = confusion_matrix(labels, y_pred)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax)
        st.pyplot(fig)
        render_classification_report(labels, y_pred, target_names=classes)

    # Full Pipeline
    st.subheader("🔍 Full Pipeline (Organ → Diagnosis)")
    if st.button("Run Full Pipeline"):

        model_paths = {
            "organ": os.path.join(PROJ_ROOT, "models", "organ_classifier.h5"),
            "kidney": os.path.join(PROJ_ROOT, "models", "kidney_diagnosis.h5"),
            "liver": os.path.join(PROJ_ROOT, "models", "liver_diagnosis.h5")
        }
        df, summary = run_full_pipeline(DATA_VAL, model_paths)

        st.markdown("### 🔬 Per Image Predictions")
        st.dataframe(df, use_container_width=True)

        st.markdown("### 📊 Summary")
        for k, v in summary.items():
            st.markdown(f"- **{k}:** {v}")

# --- Tab 3: Predict New Image ---
with tab3:
    st.header("📂 Predict New Image Full Pipeline")
    uploaded = st.file_uploader("Upload an ultrasound image", type=['png','jpg','jpeg'])
    if uploaded:
        data = uploaded.getvalue()
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        tmp_path = os.path.join(PROJ_ROOT, 'temp_upload.png')
        cv2.imwrite(tmp_path, img)
        result = subprocess.run(
            [PYTHON, os.path.join(SRC_DIR, 'predict_full_pipeline.py'), tmp_path],
            text=True,
            capture_output=True
        )
        st.code(result.stdout)

        if result.stderr:
            st.error("⚠️ Error:")
            st.code(result.stderr)

