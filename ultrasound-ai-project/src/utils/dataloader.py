import os
import numpy as np
import cv2
import random
import tensorflow as tf
from collections import defaultdict
from sklearn.model_selection import train_test_split
from utils.preprocessing import extract_roi

def load_data(data_dir,
              img_size=(224, 224),
              batch_size=32,
              val_split=0.2,
              seed=42):
    image_paths, labels = [], []
    class_names = []
    class_to_index = {}
    idx = 0

    # Step 1: Walk and collect full image path-label pairs
    for organ in os.listdir(data_dir):
        organ_path = os.path.join(data_dir, organ)
        if not os.path.isdir(organ_path):
            continue

        for label_name in os.listdir(organ_path):
            label_path = os.path.join(organ_path, label_name)
            if not os.path.isdir(label_path):
                continue

            label_full = f"{organ}_{label_name}"  # e.g., kidney_stone
            if label_full not in class_to_index:
                class_to_index[label_full] = idx
                class_names.append(label_full)
                idx += 1

            for fname in os.listdir(label_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(label_path, fname)
                    image_paths.append(full_path)
                    labels.append(class_to_index[label_full])

    print(f"📊 Found {len(image_paths)} images in {len(class_names)} classes")

    # Step 2: Balance dataset: limit kidney count to liver count
    grouped = defaultdict(list)
    for path, label in zip(image_paths, labels):
        key = "kidney" if "kidney" in path.replace("\\", "/") else "liver"
        grouped[key].append((path, label))

    liver_count = len(grouped["liver"])
    random.seed(seed)
    kidney_sampled = random.sample(grouped["kidney"], min(liver_count, len(grouped["kidney"])))

    balanced_data = grouped["liver"] + kidney_sampled
    image_paths, labels = zip(*balanced_data)

    # Step 3: Split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels,
        test_size=val_split,
        stratify=labels,
        random_state=seed
    )

    def _preprocess(path, label):
        def _load_and_process(path_str):
            img = extract_roi(path_str.decode())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            return (img.astype(np.float32) / 255.0)
        image = tf.numpy_function(_load_and_process, [path], tf.float32)
        image.set_shape((img_size[0], img_size[1], 3))
        return image, label

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = (train_ds.shuffle(len(train_paths), seed=seed)
                        .map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                        .batch(batch_size)
                        .prefetch(tf.data.AUTOTUNE))

    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = (val_ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                     .batch(batch_size)
                     .prefetch(tf.data.AUTOTUNE))

    # Count samples by organ
    def count_by_prefix(paths, prefix):
        return sum(1 for p in paths if prefix in p.replace("\\", "/"))

    print(f"\n📦 Training Set: {len(train_paths)} images")
    print(f"   - Liver:  {count_by_prefix(train_paths, 'liver')}")
    print(f"   - Kidney: {count_by_prefix(train_paths, 'kidney')}")

    print(f"\n🧪 Validation Set: {len(val_paths)} images")
    print(f"   - Liver:  {count_by_prefix(val_paths, 'liver')}")
    print(f"   - Kidney: {count_by_prefix(val_paths, 'kidney')}")

    return train_ds, val_ds, class_names
