import os
import random
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils.preprocessing import extract_roi

def load_organ_data(data_dir,
                    img_size=(224,224),
                    batch_size=32,
                    val_split=0.2,
                    seed=42):
    # 1) Gather paths & organ labels
    image_paths, labels = [], []
    organs = ["kidney", "liver"]
    for i, organ in enumerate(organs):
        organ_dir = os.path.join(data_dir, organ)
        if not os.path.isdir(organ_dir): continue

        # scan all subfolders (e.g. normal, stone, benign...)
        for sub in os.listdir(organ_dir):
            sub_dir = os.path.join(organ_dir, sub)
            if not os.path.isdir(sub_dir): continue
            for fname in os.listdir(sub_dir):
                if fname.lower().endswith((".png",".jpg",".jpeg")):
                    image_paths.append(os.path.join(sub_dir, fname))
                    labels.append(i)

    print(f"📊 Total images: {len(image_paths)} | Kidney={labels.count(0)} Liver={labels.count(1)}")

    # 2) Balance: cap kidney to liver count if desired
    #    (optional—comment out if you want full data)
    idxs = list(range(len(labels)))
    kidney_idxs = [j for j in idxs if labels[j]==0]
    liver_idxs = [j for j in idxs if labels[j]==1]
    n = len(liver_idxs)
    random.seed(seed)
    sampled_k = random.sample(kidney_idxs, min(n, len(kidney_idxs)))
    keep = set(liver_idxs + sampled_k)
    image_paths = [image_paths[j] for j in keep]
    labels       = [labels[j]       for j in keep]
    print(f"⚖️  Balanced to Kidney={labels.count(0)} Liver={labels.count(1)}")

    # 3) Stratified split
    train_p, val_p, train_l, val_l = train_test_split(
        image_paths, labels,
        test_size=val_split,
        stratify=labels,
        random_state=seed
    )

    # 4) tf.data pipelines
    def _preprocess(path, label):
        def _load(path_str):
            img = extract_roi(path_str.decode())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            return img.astype(np.float32) / 255.0

        img = tf.numpy_function(_load, [path], tf.float32)
        img.set_shape((img_size[0], img_size[1], 3))
        return img, label

    train_ds = tf.data.Dataset.from_tensor_slices((train_p, train_l))
    train_ds = (train_ds.shuffle(len(train_p), seed=seed)
                       .map(_preprocess, tf.data.AUTOTUNE)
                       .batch(batch_size)
                       .prefetch(tf.data.AUTOTUNE))

    val_ds = tf.data.Dataset.from_tensor_slices((val_p, val_l))
    val_ds = (val_ds.map(_preprocess, tf.data.AUTOTUNE)
                   .batch(batch_size)
                   .prefetch(tf.data.AUTOTUNE))

    print(f"\n📦 Train: {len(train_p)} | 🧪 Val: {len(val_p)}\n")
    return train_ds, val_ds, organs

def load_kidney_data(data_dir,
                     img_size=(224,224),
                     batch_size=32,
                     val_split=0.2,
                     seed=42):
    """
    Loads only kidney images (normal vs stone), splits 80/20,
    and returns (train_ds, val_ds, class_names=["normal","stone"])
    """
    import os, random, cv2, numpy as np, tensorflow as tf
    from sklearn.model_selection import train_test_split
    from utils.preprocessing import extract_roi

    # 1) collect paths & labels
    paths, labels = [], []
    # Limit total to 1000, balance 500 normal + 500 stone if available
    max_total = 1000
    per_class_limit = max_total // 2

    for label, sub in enumerate(("normal", "stone")):
        folder = os.path.join(data_dir, "kidney", sub)
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        selected = files[:per_class_limit]  # take first N (or shuffle then slice if needed)
        for f in selected:
            paths.append(os.path.join(folder, f))
            labels.append(label)


    print(f"📊 Kidney images: normal={labels.count(0)} stone={labels.count(1)}")

    # 2) split
    if isinstance(val_split, str) and val_split == "all":
        # Use all data as validation
        train_p, train_l = [], []
        val_p, val_l = paths, labels
    elif isinstance(val_split, float) and val_split >= 1.0:
        # Also allow val_split = 1.0 safely
        train_p, train_l = [], []
        val_p, val_l = paths, labels
    else:
        train_p, val_p, train_l, val_l = train_test_split(
            paths, labels,
            test_size=val_split,
            stratify=labels,
            random_state=seed
        )

    # 3) pipeline
    def _prep(path, label):
        def _load(p):
            img = extract_roi(p.decode())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            return img.astype(np.float32)/255.0
        img = tf.numpy_function(_load, [path], tf.float32)
        img.set_shape((*img_size,3))
        return img, label

    if len(train_p) > 0:
        train_ds = (tf.data.Dataset
                    .from_tensor_slices((train_p, train_l))
                    .shuffle(len(train_p), seed)
                    .map(_prep, tf.data.AUTOTUNE)
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE))
    else:
        train_ds = tf.data.Dataset.from_tensor_slices(([], [])).batch(1)

    val_ds = (tf.data.Dataset
                  .from_tensor_slices((val_p, val_l))
                  .map(_prep, tf.data.AUTOTUNE)
                  .batch(batch_size)
                  .prefetch(tf.data.AUTOTUNE))

    print(f"📦 Kidney train={len(train_p)} | 🧪 val={len(val_p)}")
    return train_ds, val_ds, ["normal","stone"]

def load_liver_data(data_dir,
                    img_size=(224,224),
                    batch_size=32,
                    val_split=0.2,
                    seed=42):
    """
    Loads only liver images (normal vs benign vs malignant),
    splits according to val_split, and returns
    (train_ds, val_ds, class_names=["normal","benign","malignant"])
    """
    import os, random, cv2, numpy as np, tensorflow as tf
    from sklearn.model_selection import train_test_split
    from utils.preprocessing import extract_roi

    # 1) collect paths & labels
    class_names = ["normal","benign","malignant"]
    paths, labels = [], []
    for label, sub in enumerate(class_names):
        folder = os.path.join(data_dir, "liver", sub)
        for f in os.listdir(folder):
            if f.lower().endswith((".png",".jpg",".jpeg")):
                paths.append(os.path.join(folder, f))
                labels.append(label)

    print(f"📊 Liver images: " +
          f"normal={labels.count(0)} " +
          f"benign={labels.count(1)} " +
          f"malignant={labels.count(2)}")

    # 2) split
    if isinstance(val_split, str) and val_split == "all":
        train_p, train_l = [], []
        val_p, val_l = paths, labels
    elif isinstance(val_split, float) and val_split >= 1.0:
        train_p, train_l = [], []
        val_p, val_l = paths, labels
    else:
        train_p, val_p, train_l, val_l = train_test_split(
            paths, labels,
            test_size=val_split,
            stratify=labels,
            random_state=seed
        )

    # 3) pipeline
    def _prep(path, label):
        def _load(p):
            img = extract_roi(p.decode())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            return img.astype(np.float32)/255.0

        img = tf.numpy_function(_load, [path], tf.float32)
        img.set_shape((*img_size,3))
        return img, label

    if len(train_p) > 0:
        train_ds = (tf.data.Dataset
                    .from_tensor_slices((train_p, train_l))
                    .shuffle(len(train_p), seed)
                    .map(_prep, tf.data.AUTOTUNE)
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE))
    else:
        train_ds = tf.data.Dataset.from_tensor_slices(([], [])).batch(1)

    val_ds = (tf.data.Dataset
                  .from_tensor_slices((val_p, val_l))
                  .map(_prep, tf.data.AUTOTUNE)
                  .batch(batch_size)
                  .prefetch(tf.data.AUTOTUNE))

    print(f"📦 Liver train={len(train_p)} | 🧪 val={len(val_p)}")
    return train_ds, val_ds, class_names
