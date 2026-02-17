"""
Comprehensive training script for lung nodule detection with accuracy tracking and visualization.
"""
import os
import sys
import numpy as np
from pathlib import Path

# Check if we have the required dependencies
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("ERROR: PyTorch is not installed. Please install it with: pip install torch torchvision")
    sys.exit(1)

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    print("WARNING: SimpleITK not available. Will use synthetic data or numpy arrays only.")
    SITK_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("WARNING: Matplotlib not available. Plots will not be generated.")
    MATPLOTLIB_AVAILABLE = False

from src.train import train
from sklearn.model_selection import train_test_split


def generate_synthetic_data(num_samples=200, image_size=256):
    """
    Generate synthetic CT scan-like images for training when real data is unavailable.
    Improved version with more realistic nodule patterns.
    
    Args:
        num_samples: Number of synthetic images to generate
        image_size: Size of each image (image_size x image_size)
    
    Returns:
        images: List of numpy arrays
        labels: List of binary labels (0 or 1)
    """
    print(f"Generating {num_samples} synthetic CT scan images...")
    images = []
    labels = []
    
    np.random.seed(42)  # For reproducibility
    
    for i in range(num_samples):
        # Create a base image with noise (simulating CT scan)
        img = np.random.normal(0.5, 0.2, (image_size, image_size))
        img = np.clip(img, 0, 1)
        
        # Add some structure (simulating lung tissue)
        x, y = np.meshgrid(np.linspace(0, 1, image_size), np.linspace(0, 1, image_size))
        img += 0.1 * np.sin(10 * x) * np.cos(10 * y)
        img = np.clip(img, 0, 1)
        
        # Randomly add a "nodule" (circular bright spot) to some images
        has_nodule = np.random.random() > 0.5
        label = 1.0 if has_nodule else 0.0
        
        if has_nodule:
            # Add a circular nodule with varying intensity
            center_x = np.random.randint(image_size // 4, 3 * image_size // 4)
            center_y = np.random.randint(image_size // 4, 3 * image_size // 4)
            radius = np.random.randint(8, 20)
            intensity = np.random.uniform(0.3, 0.6)
            
            y_coords, x_coords = np.ogrid[:image_size, :image_size]
            mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= radius**2
            
            # Create gradient effect for more realistic nodule
            dist = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            gradient = np.exp(-dist / (radius * 0.7))
            gradient = np.clip(gradient, 0, 1)
            
            img[mask] = np.clip(img[mask] + intensity * gradient[mask], 0, 1)
        
        images.append(img.astype(np.float32))
        labels.append(label)
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{num_samples} images...")
    
    print(f"Generated {len(images)} synthetic images ({sum(labels):.0f} with nodules, {len(labels) - sum(labels):.0f} without)")
    return images, labels


def load_luna16_data(data_dir="luna datasets", annotations_csv=None, max_images=200):
    """
    Load LUNA16 dataset if available.
    
    Args:
        data_dir: Directory containing LUNA16 data
        annotations_csv: Path to annotations CSV file
        max_images: Maximum number of images to load
    
    Returns:
        images: List of image arrays
        labels: List of labels (if annotations available)
    """
    import glob
    import pandas as pd
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Data directory '{data_dir}' not found.")
        return None, None
    
    images = []
    labels = []
    nodule_series = set()
    
    # Load annotations if available
    if annotations_csv:
        csv_path = Path(annotations_csv)
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if 'seriesuid' in df.columns:
                    nodule_series = set(df['seriesuid'])
                    print(f"Loaded {len(nodule_series)} nodule series from annotations")
            except Exception as e:
                print(f"Warning: Could not load annotations: {e}")
    
    # Find .mhd files
    mhd_files = list(data_path.rglob('*.mhd'))
    
    if not mhd_files:
        print(f"No .mhd files found in '{data_dir}'")
        return None, None
    
    print(f"Found {len(mhd_files)} .mhd files. Loading up to {max_images}...")
    
    loaded_count = 0
    for mhd_path in mhd_files[:max_images]:
        try:
            if not SITK_AVAILABLE:
                print("SimpleITK not available. Cannot load .mhd files.")
                return None, None
            
            itk_img = sitk.ReadImage(str(mhd_path))
            img_array = sitk.GetArrayFromImage(itk_img)  # [slices, h, w]
            
            # Extract central slice
            central_slice = img_array[img_array.shape[0] // 2]
            
            # Resize to 256x256
            from PIL import Image
            pil_img = Image.fromarray(central_slice)
            pil_img = pil_img.resize((256, 256), resample=Image.BILINEAR)
            arr = np.array(pil_img).astype(np.float32)
            
            # Normalize to [0, 1]
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max > arr_min:
                arr = (arr - arr_min) / (arr_max - arr_min)
            
            images.append(arr)
            
            # Label based on annotations
            if nodule_series:
                seriesuid = mhd_path.stem
                label = 1.0 if seriesuid in nodule_series else 0.0
            else:
                label = 0.0  # Unknown
            
            labels.append(label)
            loaded_count += 1
            
            if loaded_count % 10 == 0:
                print(f"  Loaded {loaded_count}/{min(len(mhd_files), max_images)} images...")
                
        except Exception as e:
            print(f"Failed to load {mhd_path}: {e}")
            continue
    
    if images:
        print(f"Successfully loaded {len(images)} images from LUNA16 dataset")
        return images, labels if nodule_series else None
    else:
        return None, None


def print_final_metrics(history):
    """Print final training metrics summary."""
    print("\n" + "=" * 60)
    print("FINAL TRAINING METRICS")
    print("=" * 60)
    
    if history['train_accuracy']:
        print(f"\nTraining Metrics:")
        print(f"  Final Loss: {history['train_loss'][-1]:.4f}")
        print(f"  Final Accuracy: {history['train_accuracy'][-1]:.4f} ({history['train_accuracy'][-1]*100:.2f}%)")
        print(f"  Final Precision: {history['train_precision'][-1]:.4f}")
        print(f"  Final Recall: {history['train_recall'][-1]:.4f}")
        print(f"  Final F1 Score: {history['train_f1'][-1]:.4f}")
        
        print(f"\nBest Training Metrics:")
        print(f"  Best Accuracy: {max(history['train_accuracy']):.4f} ({max(history['train_accuracy'])*100:.2f}%)")
        print(f"  Best F1 Score: {max(history['train_f1']):.4f}")
    
    if history['val_accuracy']:
        print(f"\nValidation Metrics:")
        print(f"  Final Loss: {history['val_loss'][-1]:.4f}")
        print(f"  Final Accuracy: {history['val_accuracy'][-1]:.4f} ({history['val_accuracy'][-1]*100:.2f}%)")
        print(f"  Final Precision: {history['val_precision'][-1]:.4f}")
        print(f"  Final Recall: {history['val_recall'][-1]:.4f}")
        print(f"  Final F1 Score: {history['val_f1'][-1]:.4f}")
        
        print(f"\nBest Validation Metrics:")
        print(f"  Best Accuracy: {max(history['val_accuracy']):.4f} ({max(history['val_accuracy'])*100:.2f}%)")
        print(f"  Best F1 Score: {max(history['val_f1']):.4f}")
        print(f"  Best Loss: {min(history['val_loss']):.4f}")
    
    print("=" * 60)


def main():
    """Main training function."""
    print("=" * 60)
    print("Lung Nodule Detection - Training Script with Accuracy Tracking")
    print("=" * 60)
    print()
    
    # Configuration
    config = {
        'data_dir': 'luna datasets',
        'annotations_csv': None,
        'max_images': 200,
        'epochs': 30,  # Increased for better training
        'batch_size': 16,  # Increased batch size
        'learning_rate': 0.001,
        'use_synthetic': False,
        'synthetic_samples': 200,  # Increased for better training
        'model_save_path': 'models/trained_model.pth',
        'use_improved_model': True  # Use improved CNN architecture
    }
    
    # Try to load real data
    print("Attempting to load LUNA16 dataset...")
    images, labels = load_luna16_data(
        data_dir=config['data_dir'],
        annotations_csv=config['annotations_csv'] or os.path.join(config['data_dir'], 'annotations.csv'),
        max_images=config['max_images']
    )
    
    # Fallback to synthetic data if no real data available
    if images is None or len(images) == 0:
        print("\nNo real data found. Generating synthetic data for training...")
        images, labels = generate_synthetic_data(num_samples=config['synthetic_samples'])
        config['use_synthetic'] = True
    
    if len(images) == 0:
        print("ERROR: No data available for training!")
        return
    
    print(f"\nDataset Summary:")
    print(f"  Total images: {len(images)}")
    if labels:
        print(f"  Positive samples: {sum(labels):.0f}")
        print(f"  Negative samples: {len(labels) - sum(labels):.0f}")
        print(f"  Class balance: {sum(labels)/len(labels)*100:.1f}% positive")
    print()
    
    # Split into train/validation sets
    if labels:
        print("Splitting into train/validation sets (80/20)...")
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                images, labels, 
                test_size=0.2, 
                random_state=42, 
                stratify=labels
            )
            print(f"  Train: {len(X_train)} images ({sum(y_train):.0f} positive, {len(y_train)-sum(y_train):.0f} negative)")
            print(f"  Validation: {len(X_val)} images ({sum(y_val):.0f} positive, {len(y_val)-sum(y_val):.0f} negative)")
        except ValueError as e:
            print(f"Warning: Could not stratify split: {e}")
            print("Using random split instead...")
            X_train, X_val, y_train, y_val = train_test_split(
                images, labels, 
                test_size=0.2, 
                random_state=42
            )
    else:
        print("No labels available. Using all data for training...")
        X_train = images
        X_val = None
        y_train = None
        y_val = None
    
    print()
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Start training
    print("Starting training...")
    print(f"  Model: {'ImprovedCNN' if config['use_improved_model'] else 'SimpleCNN'}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print()
    
    try:
        model, history = train(
            images=X_train,
            labels=y_train,
            val_images=X_val,
            val_labels=y_val,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            model_save_path=config['model_save_path'],
            use_improved_model=config['use_improved_model'],
            plot_history=True
        )
        
        if model:
            print("\n" + "=" * 60)
            print("Training completed successfully!")
            print(f"Model saved to: {config['model_save_path']}")
            
            # Print final metrics
            print_final_metrics(history)
            
            # Check if plots were generated
            plot_path = config['model_save_path'].replace('.pth', '_training_history.png')
            if os.path.exists(plot_path):
                print(f"\nTraining plots saved to: {plot_path}")
            
            print("=" * 60)
        else:
            print("\nTraining completed but model was not returned.")
            
    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
