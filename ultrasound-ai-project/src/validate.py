import os
from keras.models import load_model
from utils.dataloader import load_organ_data

def main():
    root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(root, "data")
    _, val_ds, organs = load_organ_data(data_dir)

    model = load_model(os.path.join(root, "models", "organ_classifier.h5"))
    loss, acc = model.evaluate(val_ds)
    print(f"Validation accuracy (kidney vs liver): {acc*100:.2f}%")

if __name__=="__main__":
    main()
