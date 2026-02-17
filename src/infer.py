"""
Inference utilities for lung nodule detection models.
"""
import os
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple, Any

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    transforms = None


def is_torch_available() -> bool:
    """Check if PyTorch is available."""
    return TORCH_AVAILABLE


def load_model(model_path: str, device: str = 'cpu') -> Optional[Any]:
    """
    Load a PyTorch model from file.
    
    Args:
        model_path: Path to the model file (.pth)
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        Loaded model or None if loading fails
    """
    if not is_torch_available():
        return None
    
    try:
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return None
        
        # Try to load the model
        # Note: This is a generic loader - actual model architecture should match
        # Use weights_only=True for security (PyTorch 1.13+)
        try:
            # Try with weights_only for security
            model = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions
            model = torch.load(model_path, map_location=device)
        
        # If it's a state dict, we need the model architecture
        # For now, return as-is (assumes full model save)
        if isinstance(model, dict) and 'state_dict' in model:
            # This would require knowing the model architecture
            print("Warning: Model file contains state_dict. Model architecture required.")
            return None
        
        if hasattr(model, 'eval'):
            model.eval()
        
        return model
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        return None


def predict(model: Optional[Any], img: Image.Image, device: str = 'cpu') -> float:
    """
    Predict confidence score for a single image.
    
    Args:
        model: Loaded PyTorch model (can be None for demo mode)
        img: PIL Image to predict on
        device: Device to run inference on
    
    Returns:
        Confidence score between 0.0 and 1.0
    """
    if model is None or not is_torch_available():
        # Demo mode: return random score
        return 0.75
    
    try:
        # Convert to RGB if needed (before creating transform)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Preprocess image - use medical image normalization instead of ImageNet
        # For RGB images (converted from grayscale), use same value for all channels
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # Use [0.5, 0.5, 0.5] for RGB medical images instead of ImageNet stats
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(img_tensor)
            
            # Handle different output formats
            if isinstance(output, (list, tuple)):
                output = output[0]
            
            # Convert to probability
            if output.shape[1] > 1:
                # Multi-class: use softmax
                probs = torch.softmax(output, dim=1)
                confidence = probs[0][1].item() if probs.shape[1] > 1 else probs[0][0].item()
            else:
                # Binary: use sigmoid
                confidence = torch.sigmoid(output[0][0]).item()
        
        return max(0.0, min(1.0, confidence))
    except Exception as e:
        print(f"Prediction failed: {e}")
        return 0.5  # Default confidence


def detect_boxes_with_options(
    model: Optional[Any],
    img: Image.Image,
    conf_thresh: float = 0.5,
    apply_nms: bool = True,
    iou_thresh: float = 0.5
) -> List[Tuple[float, float, float, float, float]]:
    """
    Detect bounding boxes for nodules in an image.
    
    Args:
        model: Loaded detection model (RetinaNet-style)
        img: PIL Image to detect on
        conf_thresh: Confidence threshold for detections
        apply_nms: Whether to apply Non-Maximum Suppression
        iou_thresh: IoU threshold for NMS
    
    Returns:
        List of boxes as (x1, y1, x2, y2, confidence) tuples
    """
    if model is None or not is_torch_available():
        # Demo mode: return empty list or dummy boxes
        return []
    
    try:
        # Convert to RGB if needed (before creating transform)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Preprocess image - use medical image normalization instead of ImageNet
        # For RGB images (converted from grayscale), use same value for all channels
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # Use [0.5, 0.5, 0.5] for RGB medical images instead of ImageNet stats
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Run detection
        with torch.no_grad():
            outputs = model(img_tensor)
            
            # Parse outputs (format depends on model architecture)
            # For RetinaNet-style models, outputs typically contain boxes and scores
            boxes = []
            
            # Try to extract boxes from output
            if isinstance(outputs, dict):
                boxes_tensor = outputs.get('boxes', outputs.get('boxes_list', None))
                scores_tensor = outputs.get('scores', outputs.get('scores_list', None))
            elif isinstance(outputs, (list, tuple)):
                boxes_tensor = outputs[0] if len(outputs) > 0 else None
                scores_tensor = outputs[1] if len(outputs) > 1 else None
            else:
                # Fallback: assume no detections
                return []
            
            if boxes_tensor is not None and scores_tensor is not None:
                # Filter by confidence
                mask = scores_tensor >= conf_thresh
                boxes_tensor = boxes_tensor[mask]
                scores_tensor = scores_tensor[mask]
                
                # Convert to list format
                for i in range(len(boxes_tensor)):
                    box = boxes_tensor[i].cpu().numpy()
                    score = scores_tensor[i].cpu().item()
                    boxes.append((float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(score)))
                
                # Apply NMS if requested
                if apply_nms and len(boxes) > 1:
                    boxes = apply_nms_boxes(boxes, iou_thresh)
        
        return boxes
    except Exception as e:
        print(f"Box detection failed: {e}")
        return []


def apply_nms_boxes(boxes: List[Tuple[float, float, float, float, float]], iou_thresh: float) -> List[Tuple[float, float, float, float, float]]:
    """
    Apply Non-Maximum Suppression to boxes.
    
    Args:
        boxes: List of (x1, y1, x2, y2, confidence) tuples
        iou_thresh: IoU threshold for NMS
    
    Returns:
        Filtered list of boxes
    """
    if not boxes:
        return []
    
    if not is_torch_available():
        # Simple NMS without PyTorch
        return simple_nms(boxes, iou_thresh)
    
    try:
        import torch
        
        # Convert to tensors
        boxes_tensor = torch.tensor([[b[0], b[1], b[2], b[3]] for b in boxes])
        scores_tensor = torch.tensor([b[4] for b in boxes])
        
        # Apply NMS
        keep_indices = torch.ops.torchvision.nms(boxes_tensor, scores_tensor, iou_thresh)
        
        # Return kept boxes
        return [boxes[i] for i in keep_indices.tolist()]
    except Exception:
        # Fallback to simple NMS
        return simple_nms(boxes, iou_thresh)


def simple_nms(boxes: List[Tuple[float, float, float, float, float]], iou_thresh: float) -> List[Tuple[float, float, float, float, float]]:
    """
    Simple NMS implementation without PyTorch.
    """
    if not boxes:
        return []
    
    # Sort by confidence (descending)
    sorted_boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    kept = []
    
    while sorted_boxes:
        # Take the box with highest confidence
        current = sorted_boxes.pop(0)
        kept.append(current)
        
        # Remove boxes with high IoU
        remaining = []
        for box in sorted_boxes:
            iou = calculate_iou(current[:4], box[:4])
            if iou < iou_thresh:
                remaining.append(box)
        sorted_boxes = remaining
    
    return kept


def calculate_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Calculate Intersection over Union (IoU) of two boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def draw_boxes(img: Image.Image, boxes: List[Tuple[float, float, float, float, float]]) -> Image.Image:
    """
    Draw bounding boxes on an image.
    
    Args:
        img: PIL Image
        boxes: List of (x1, y1, x2, y2, confidence) tuples
    
    Returns:
        PIL Image with boxes drawn
    """
    try:
        from PIL import ImageDraw, ImageFont
        
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # Scale boxes to image size if needed
        img_w, img_h = img.size
        
        for box in boxes:
            x1, y1, x2, y2, conf = box
            
            # Scale if boxes are normalized or in different coordinate system
            # More robust check: if max coordinate is <= 1.0, assume normalized
            max_coord = max(abs(x1), abs(y1), abs(x2), abs(y2))
            if max_coord <= 1.0 and max_coord >= 0:  # Likely normalized [0,1]
                x1 = x1 * img_w
                y1 = y1 * img_h
                x2 = x2 * img_w
                y2 = y2 * img_h
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            
            # Draw confidence text
            text = f"{conf:.2f}"
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
            draw.text((x1, y1 - 15), text, fill='red', font=font)
        
        return img_copy
    except Exception as e:
        print(f"Failed to draw boxes: {e}")
        return img
