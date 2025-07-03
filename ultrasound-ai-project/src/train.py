import os
import tensorflow as tf
from utils.dataloader import load_data
from utils.model_builder import build_model

def main():
    # assume you run this from the inner ultrasound-ai-project/ folder
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    train_ds, val_ds, class_names = load_data(data_dir)

    model = build_model(num_classes=len(class_names))

    # callbacks
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    cbs = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        tf.keras.callbacks.EarlyStopping(patience=5,
                                         restore_best_weights=True)
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=cbs
    )

    # save
    os.makedirs("models", exist_ok=True)
    model.save("models/organ_classifier.h5")
    print("✅ Training complete and model saved.")

if __name__ == "__main__":
    main()
