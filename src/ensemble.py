"""
Ensemble prediction utilities for combining multiple models.
"""
import os
import glob
import numpy as np
from PIL import Image
from typing import List, Optional
from . import infer


def get_model_paths(models_dir: str, extension: str = '.pth') -> List[str]:
    """
    Get all model file paths from a directory.
    
    Args:
        models_dir: Directory containing model files
        extension: File extension to search for (default: .pth)
    
    Returns:
        List of model file paths, sorted by filename
    """
    if not os.path.isdir(models_dir):
        return []
    
    pattern = os.path.join(models_dir, f'*{extension}')
    model_paths = glob.glob(pattern)
    
    # Filter out empty files
    valid_paths = []
    for path in model_paths:
        try:
            if os.path.getsize(path) > 0:
                valid_paths.append(path)
        except Exception:
            continue
    
    # Sort by filename
    valid_paths.sort(key=lambda x: os.path.basename(x))
    
    return valid_paths


def predict_ensemble(models_dir: str, img: Image.Image, device: str = 'cpu') -> float:
    """
    Predict using an ensemble of models.
    
    Averages predictions from all available models in the directory.
    
    Args:
        models_dir: Directory containing model files
        img: PIL Image to predict on
        device: Device to run inference on
    
    Returns:
        Average confidence score from all models
    """
    model_paths = get_model_paths(models_dir)
    
    if not model_paths:
        # No models available, return default score
        return 0.5
    
    predictions = []
    
    for model_path in model_paths:
        try:
            model = infer.load_model(model_path, device=device)
            if model is not None:
                score = infer.predict(model, img, device=device)
                predictions.append(score)
        except Exception as e:
            print(f"Failed to predict with model {model_path}: {e}")
            continue
    
    if not predictions:
        # No successful predictions
        return 0.5
    
    # Return average of all predictions
    return float(np.mean(predictions))


def predict_ensemble_weighted(
    models_dir: str,
    img: Image.Image,
    weights: Optional[List[float]] = None,
    device: str = 'cpu'
) -> float:
    """
    Predict using a weighted ensemble of models.
    
    Args:
        models_dir: Directory containing model files
        img: PIL Image to predict on
        weights: Optional list of weights for each model (must match number of models)
        device: Device to run inference on
    
    Returns:
        Weighted average confidence score
    """
    model_paths = get_model_paths(models_dir)
    
    if not model_paths:
        return 0.5
    
    predictions = []
    valid_models = []
    
    for model_path in model_paths:
        try:
            model = infer.load_model(model_path, device=device)
            if model is not None:
                score = infer.predict(model, img, device=device)
                predictions.append(score)
                valid_models.append(model_path)
        except Exception as e:
            print(f"Failed to predict with model {model_path}: {e}")
            continue
    
    if not predictions:
        return 0.5
    
    # Apply weights if provided
    if weights is not None:
        if len(weights) != len(predictions):
            print(f"Warning: Number of weights ({len(weights)}) doesn't match number of models ({len(predictions)}). Using equal weights.")
            weights = None
    
    if weights is not None:
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        return float(np.average(predictions, weights=weights))
    else:
        return float(np.mean(predictions))
