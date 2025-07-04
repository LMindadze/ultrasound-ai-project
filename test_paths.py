import os

root = "data"
count = 0

for organ in os.listdir(root):
    organ_path = os.path.join(root, organ)
    print(f"\n🔍 Organ: {organ_path}")

    for cls in os.listdir(organ_path):
        class_path = os.path.join(organ_path, cls)
        print(f"  📂 Class: {class_path}")

        for fname in os.listdir(class_path):
            print(f"    📄 File: {fname}")
            count += 1

print(f"\n✅ Total images found: {count}")
