"""
Image preprocessing utilities for CT scans.
"""
import numpy as np
from typing import Union, Optional


def preprocess_scan(scan: np.ndarray, window_center: int = -600, window_width: int = 1500) -> np.ndarray:
    """
    Preprocess a CT scan for model input.
    
    Steps:
    1. Apply windowing (HU to grayscale)
    2. Normalize to [0, 1] range
    3. Optionally resize
    
    Args:
        scan: CT scan array of shape (slices, H, W) or (H, W)
        window_center: Window center in HU (default: -600 for lung window)
        window_width: Window width in HU (default: 1500 for lung window)
    
    Returns:
        Preprocessed scan array normalized to [0, 1] range
    
    Raises:
        ValueError: If scan contains invalid values (NaN, Inf) or is empty
    """
    # Validate input
    if scan.size == 0:
        raise ValueError("Scan is empty")
    
    # Check for NaN or Inf values
    if np.any(np.isnan(scan)) or np.any(np.isinf(scan)):
        raise ValueError("Scan contains NaN or Inf values. Please check input data.")
    
    # Ensure float32
    scan = scan.astype(np.float32)
    
    # Apply windowing
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2
    
    scan = np.clip(scan, window_min, window_max)
    
    # Normalize to [0, 1]
    scan = (scan - window_min) / (window_max - window_min)
    
    return scan


def normalize_slice(slice_data: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize a single slice.
    
    Args:
        slice_data: 2D array (H, W)
        method: Normalization method ('minmax', 'zscore', 'percentile')
    
    Returns:
        Normalized slice
    """
    slice_data = slice_data.astype(np.float32)
    
    if method == 'minmax':
        min_val = np.min(slice_data)
        max_val = np.max(slice_data)
        if max_val > min_val:
            return (slice_data - min_val) / (max_val - min_val)
        else:
            return slice_data
    elif method == 'zscore':
        mean = np.mean(slice_data)
        std = np.std(slice_data)
        if std > 0:
            return (slice_data - mean) / std
        else:
            return slice_data
    elif method == 'percentile':
        p2 = np.percentile(slice_data, 2)
        p98 = np.percentile(slice_data, 98)
        if p98 > p2:
            return np.clip((slice_data - p2) / (p98 - p2), 0, 1)
        else:
            return slice_data
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def resize_scan(scan: np.ndarray, target_size: tuple = (256, 256)) -> np.ndarray:
    """
    Resize scan slices to target size.
    
    Args:
        scan: CT scan array of shape (slices, H, W) or (H, W)
        target_size: Target (height, width)
    
    Returns:
        Resized scan
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL/Pillow is required for resizing. Install with: pip install Pillow")
    
    if len(scan.shape) == 2:
        # Single slice
        img = Image.fromarray(scan)
        img_resized = img.resize(target_size[::-1], resample=Image.BILINEAR)  # PIL uses (W, H)
        return np.array(img_resized)
    elif len(scan.shape) == 3:
        # Multiple slices
        resized_slices = []
        for i in range(scan.shape[0]):
            slice_resized = resize_scan(scan[i], target_size)
            resized_slices.append(slice_resized)
        return np.array(resized_slices)
    else:
        raise ValueError(f"Unsupported scan shape: {scan.shape}")


def extract_central_slice(scan: np.ndarray) -> np.ndarray:
    """
    Extract the central slice from a 3D scan.
    
    Args:
        scan: CT scan array of shape (slices, H, W)
    
    Returns:
        Central slice of shape (H, W)
    """
    if len(scan.shape) != 3:
        raise ValueError(f"Expected 3D scan, got shape: {scan.shape}")
    
    central_idx = scan.shape[0] // 2
    return scan[central_idx]
