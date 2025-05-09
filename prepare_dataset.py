import os
import shutil

# Get the absolute base path where this script is running
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
OUTPUT_PATH = os.path.join(BASE_DIR, "TeachableDataset")
NORMAL_PATH = os.path.join(OUTPUT_PATH, "Normal")
ANOMALY_PATH = os.path.join(OUTPUT_PATH, "Anomaly")

# Good Cap class index
GOOD_CAP_CLASS_INDEX = 2

# Create output dirs
os.makedirs(NORMAL_PATH, exist_ok=True)
os.makedirs(ANOMALY_PATH, exist_ok=True)

# Process each split
for split in ['train', 'valid', 'test']:
    image_dir = os.path.join(DATASET_PATH, split, 'images')
    label_dir = os.path.join(DATASET_PATH, split, 'labels')

    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        print(f"⚠️ Skipping {split}: folders missing")
        continue

    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue

        label_path = os.path.join(label_dir, label_file)
        image_name = os.path.splitext(label_file)[0] + '.jpg'
        image_path = os.path.join(image_dir, image_name)

        if not os.path.exists(image_path):
            continue

        with open(label_path, 'r') as f:
            contents = f.read().strip()
            if contents == '':
                continue
            classes = [int(line.split()[0]) for line in contents.splitlines()]
            if all(cls == GOOD_CAP_CLASS_INDEX for cls in classes):
                dest = NORMAL_PATH
            else:
                dest = ANOMALY_PATH

        shutil.copy(image_path, os.path.join(dest, image_name))

print(f"✅ Teachable Machine dataset prepared.\nNormal: {len(os.listdir(NORMAL_PATH))} images\nAnomaly: {len(os.listdir(ANOMALY_PATH))} images")
