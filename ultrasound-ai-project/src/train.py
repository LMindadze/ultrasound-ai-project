import os
import tensorflow as tf
from utils.dataloader import load_organ_data
from utils.model_builder import build_organ_model

def main():
    root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(root, "data")
    train_ds, val_ds, organs = load_organ_data(data_dir)

    model = build_organ_model(num_classes=len(organs))

    cbs = [
        tf.keras.callbacks.TensorBoard(log_dir="logs"),
        tf.keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=cbs)
    model.save(os.path.join(root, "models", "organ_classifier.h5"))
    print("✅ Stage 1 model trained & saved to models/organ_classifier.h5")

if __name__=="__main__":
    main()
