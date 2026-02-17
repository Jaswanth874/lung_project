import os
import glob
import SimpleITK as sitk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.train import train

def load_luna16_images_and_labels(data_dir, annotations_csv, max_images=200):
    # Load annotations
    df = pd.read_csv(annotations_csv)
    nodule_series = set(df['seriesuid'])
    # Find all .mhd files in the dataset (limit for demo)
    mhd_files = glob.glob(os.path.join(data_dir, '**', '*.mhd'), recursive=True)
    images, labels = [], []
    for i, mhd_path in enumerate(mhd_files):
        if i >= max_images:
            break
        try:
            seriesuid = os.path.splitext(os.path.basename(mhd_path))[0]
            itk_img = sitk.ReadImage(mhd_path)
            img_array = sitk.GetArrayFromImage(itk_img)  # [slices, h, w]
            central_slice = img_array[img_array.shape[0] // 2]
            from PIL import Image
            pil_img = Image.fromarray(central_slice)
            pil_img = pil_img.resize((256, 256), resample=Image.BILINEAR)
            arr = np.array(pil_img).astype(np.float32)
            images.append(arr)
            # Label: 1 if nodule present, else 0
            label = 1.0 if seriesuid in nodule_series else 0.0
            labels.append(label)
            print(f"Loaded {mhd_path} (label={label})")
        except Exception as e:
            print(f"Failed to load {mhd_path}: {e}")
    return images, labels

if __name__ == "__main__":
    data_dir = r"luna datasets"
    annotations_csv = os.path.join(data_dir, "annotations.csv")
    images, labels = load_luna16_images_and_labels(data_dir, annotations_csv, max_images=200)
    print(f"Loaded {len(images)} images. Splitting train/val...")
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    train(X_train, y_train, X_val, y_val, epochs=20)
    print("Training complete.")
