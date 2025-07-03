import os, tensorflow as tf
from utils.dataloader import load_kidney_data
from utils.model_builder import build_organ_model  # reuse binary builder

def main():
    root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(root, "data")
    train_ds, val_ds, classes = load_kidney_data(data_dir)

    model = build_organ_model(num_classes=len(classes))

    cbs = [
      tf.keras.callbacks.TensorBoard(log_dir="logs/kidney"),
      tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=cbs)
    os.makedirs(os.path.join(root,"models"), exist_ok=True)
    model.save(os.path.join(root, "models", "kidney_diagnosis.h5"))
    print("✅ Kidney model saved to models/kidney_diagnosis.h5")

if __name__=="__main__":
    main()
