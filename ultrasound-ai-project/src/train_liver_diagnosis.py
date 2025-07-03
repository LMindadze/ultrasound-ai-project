import os, tensorflow as tf
from utils.dataloader import load_liver_data
from utils.model_builder import build_organ_model

def main():
    root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(root, "data")

    train_ds, val_ds, classes = load_liver_data(
        data_dir,
        img_size=(224,224),
        batch_size=32,
        val_split=0.2,
        seed=42
    )

    model = build_organ_model(num_classes=len(classes))

    cbs = [
      tf.keras.callbacks.TensorBoard(log_dir="logs/liver"),
      tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=20,
              callbacks=cbs)

    os.makedirs(os.path.join(root,"models"), exist_ok=True)
    model.save(os.path.join(root, "models", "liver_diagnosis.h5"))
    print("✅ Liver model saved to models/liver_diagnosis.h5")

if __name__=="__main__":
    main()
