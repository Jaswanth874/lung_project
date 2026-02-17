"""
CT scan data loading utilities.
"""
import os
import numpy as np
from typing import Union, Optional

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False
    sitk = None


def load_ct_scan(file_path: str) -> np.ndarray:
    """
    Load a CT scan from file.
    
    Supports:
    - .mhd files (with .raw companion files)
    - .nii/.nii.gz files
    - NumPy arrays (.npy)
    
    Args:
        file_path: Path to the CT scan file
    
    Returns:
        NumPy array of shape (slices, height, width) or (height, width) for 2D
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CT scan file not found: {file_path}")
    
    # Check for .nii.gz first (double extension)
    if file_path.lower().endswith('.nii.gz'):
        return _load_nifti(file_path)
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.mhd':
        return _load_mhd(file_path)
    elif file_ext == '.nii':
        return _load_nifti(file_path)
    elif file_ext == '.npy':
        return np.load(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def _load_mhd(file_path: str) -> np.ndarray:
    """Load .mhd file using SimpleITK."""
    if not SITK_AVAILABLE:
        raise ImportError("SimpleITK is required to load .mhd files. Install with: pip install SimpleITK")
    
    try:
        itk_image = sitk.ReadImage(file_path)
        image_array = sitk.GetArrayFromImage(itk_image)
        
        # SimpleITK returns (z, y, x) format, which is (slices, height, width)
        return image_array
    except Exception as e:
        raise ValueError(f"Failed to load .mhd file {file_path}: {e}")


def _load_nifti(file_path: str) -> np.ndarray:
    """Load NIfTI file using SimpleITK."""
    if not SITK_AVAILABLE:
        raise ImportError("SimpleITK is required to load NIfTI files. Install with: pip install SimpleITK")
    
    try:
        itk_image = sitk.ReadImage(file_path)
        image_array = sitk.GetArrayFromImage(itk_image)
        return image_array
    except Exception as e:
        raise ValueError(f"Failed to load NIfTI file {file_path}: {e}")


def get_scan_info(scan: np.ndarray) -> dict:
    """
    Get information about a CT scan.
    
    Args:
        scan: CT scan array
    
    Returns:
        Dictionary with scan information
    """
    info = {
        'shape': scan.shape,
        'dtype': str(scan.dtype),
        'min': float(np.min(scan)),
        'max': float(np.max(scan)),
        'mean': float(np.mean(scan)),
        'std': float(np.std(scan)),
    }
    
    if len(scan.shape) == 3:
        info['num_slices'] = scan.shape[0]
        info['height'] = scan.shape[1]
        info['width'] = scan.shape[2]
    elif len(scan.shape) == 2:
        info['height'] = scan.shape[0]
        info['width'] = scan.shape[1]
    
    return info
