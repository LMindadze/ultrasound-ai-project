import os
from keras.models import load_model
from utils.dataloader import load_liver_data

def main():
    root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(root, "data")  # unseen liver data
    model = load_model(os.path.join(root, "models", "liver_diagnosis.h5"))

    # Load all liver validation data
    train_ds, val_ds, classes = load_liver_data(data_dir, val_split="all")

    loss, acc = model.evaluate(val_ds)
    print(f"Liver validation accuracy: {acc*100:.2f}%")

if __name__=="__main__":
    main()
