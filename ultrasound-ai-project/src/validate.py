import os
from keras.models import load_model
from utils.dataloader import load_data

def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    _, val_ds, class_names = load_data(data_dir)

    model = load_model("models/organ_classifier.h5")
    loss, acc = model.evaluate(val_ds)
    print(f"Validation accuracy on {len(class_names)} classes: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
