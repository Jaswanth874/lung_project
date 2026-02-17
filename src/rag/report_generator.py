"""
Alternative report generator module (for compatibility with run_pipeline.py).
"""
from .generator import generate_report as _generate_report


def create_report(confidence: float) -> str:
    """
    Create a clinical report from confidence score.
    
    This is an alias for generate_report for compatibility.
    
    Args:
        confidence: Detection confidence score (0-1)
    
    Returns:
        Generated report text
    """
    return _generate_report(confidence_score=confidence, num_detections=0)
