import os
import shutil
import random

SOURCE_DIR = "data"
TARGET_DIR = "data_validation"
SAMPLES_PER_CLASS = 6
SEED = 42

random.seed(SEED)

def split_data():
    for organ in os.listdir(SOURCE_DIR):
        organ_path = os.path.join(SOURCE_DIR, organ)
        if not os.path.isdir(organ_path):
            continue

        for cls in os.listdir(organ_path):
            cls_path = os.path.join(organ_path, cls)
            if not os.path.isdir(cls_path):
                continue

            all_files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            selected = random.sample(all_files, min(SAMPLES_PER_CLASS, len(all_files)))

            dest_dir = os.path.join(TARGET_DIR, organ, cls)
            os.makedirs(dest_dir, exist_ok=True)

            for fname in selected:
                src = os.path.join(cls_path, fname)
                dst = os.path.join(dest_dir, fname)
                shutil.move(src, dst)

    print("✅ 10 images from each class moved to `data_training/`.")

def restore_data():
    for organ in os.listdir(TARGET_DIR):
        organ_path = os.path.join(TARGET_DIR, organ)
        if not os.path.isdir(organ_path):
            continue

        for cls in os.listdir(organ_path):
            cls_path = os.path.join(organ_path, cls)
            if not os.path.isdir(cls_path):
                continue

            dest_dir = os.path.join(SOURCE_DIR, organ, cls)
            os.makedirs(dest_dir, exist_ok=True)

            for fname in os.listdir(cls_path):
                src = os.path.join(cls_path, fname)
                dst = os.path.join(dest_dir, fname)
                shutil.move(src, dst)

    print("♻️ All images restored from `data_training/` to `data/`.")

if __name__ == "__main__":
    mode = input("Type 'split' to extract samples or 'restore' to undo: ").strip().lower()

    if mode == "split":
        split_data()
    elif mode == "restore":
        restore_data()
    else:
        print("❌ Invalid option. Type 'split' or 'restore'.")
