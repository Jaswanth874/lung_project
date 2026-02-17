import os
import glob
import SimpleITK as sitk
import numpy as np
from src.train import train

def load_luna16_images(data_dir, max_images=10):
    # Find all .mhd files in the dataset (limit for demo)
    mhd_files = glob.glob(os.path.join(data_dir, '**', '*.mhd'), recursive=True)
    images = []
    for i, mhd_path in enumerate(mhd_files):
        if i >= max_images:
            break
        try:
            itk_img = sitk.ReadImage(mhd_path)
            img_array = sitk.GetArrayFromImage(itk_img)  # shape: [slices, height, width]
            # Take the central slice for simplicity
            central_slice = img_array[img_array.shape[0] // 2]
            # Resize to 256x256 to match model input
            from PIL import Image
            pil_img = Image.fromarray(central_slice)
            pil_img = pil_img.resize((256, 256), resample=Image.BILINEAR)
            arr = np.array(pil_img).astype(np.float32)
            images.append(arr)
            print(f"Loaded {mhd_path} (resized)")
        except Exception as e:
            print(f"Failed to load {mhd_path}: {e}")
    return images

if __name__ == "__main__":
    data_dir = r"luna datasets"
    images = load_luna16_images(data_dir, max_images=100)
    print(f"Loaded {len(images)} images. Starting training...")
    train(images)
    print("Training complete.")
