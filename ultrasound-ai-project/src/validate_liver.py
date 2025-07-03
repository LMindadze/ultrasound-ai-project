import os
from keras.models import load_model
from utils.dataloader import load_liver_data

def main():
    root = os.path.dirname(os.path.dirname(__file__))
    val_dir = os.path.join(root, "data")

    # Use full dataset as validation only
    _, val_ds, classes = load_liver_data(
        val_dir,
        img_size=(224, 224),
        batch_size=32,
        val_split="all",  # use all data as validation
        seed=42
    )

    model = load_model(os.path.join(root, "models", "liver_diagnosis.h5"))
    loss, acc = model.evaluate(val_ds)
    print(f"Liver validation accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
