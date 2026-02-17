"""
Knowledge retrieval for RAG-based report generation.
"""
import os
from typing import List, Dict, Optional


# Sample knowledge base for lung nodule detection
KNOWLEDGE_BASE = [
    {
        "topic": "lung_nodule_detection",
        "content": "Lung nodules are small masses of tissue in the lungs. They can be benign or malignant. Early detection is crucial for treatment success."
    },
    {
        "topic": "ct_scan_analysis",
        "content": "CT scans provide detailed cross-sectional images of the lungs. Radiologists analyze these images to identify abnormalities including nodules, masses, and other lesions."
    },
    {
        "topic": "nodule_characteristics",
        "content": "Nodules are typically classified by size, shape, and density. Suspicious nodules may have irregular borders, spiculated edges, or show growth over time."
    },
    {
        "topic": "confidence_scores",
        "content": "AI detection systems provide confidence scores ranging from 0 to 1. Higher scores indicate greater confidence in detection. Scores above 0.7 are generally considered reliable."
    },
    {
        "topic": "clinical_followup",
        "content": "Detected nodules may require follow-up imaging, biopsy, or monitoring depending on size, characteristics, and patient risk factors. Clinical correlation is essential."
    }
]


def retrieve_knowledge(
    query: str,
    confidence_score: float,
    num_detections: int = 0,
    top_k: int = 3
) -> List[Dict[str, str]]:
    """
    Retrieve relevant knowledge for report generation.
    
    Args:
        query: Search query or context
        confidence_score: Detection confidence score
        num_detections: Number of detected nodules
        top_k: Number of knowledge items to retrieve
    
    Returns:
        List of relevant knowledge items
    """
    # Simple keyword-based retrieval
    # In a production system, this would use vector embeddings and similarity search
    
    query_lower = query.lower()
    scored_items = []
    
    for item in KNOWLEDGE_BASE:
        score = 0.0
        topic = item["topic"].lower()
        content = item["content"].lower()
        
        # Score based on keyword matching
        keywords = ["nodule", "detection", "lung", "ct", "scan", "confidence", "clinical"]
        for keyword in keywords:
            if keyword in query_lower or keyword in content:
                score += 1.0
        
        # Boost score based on confidence level
        if "confidence" in topic and confidence_score > 0.5:
            score += 2.0
        
        # Boost score based on detections
        if "detection" in topic and num_detections > 0:
            score += 1.0
        
        scored_items.append((score, item))
    
    # Sort by score and return top_k
    scored_items.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored_items[:top_k]]


def load_knowledge_base(file_path: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Load knowledge base from file (if available).
    
    Args:
        file_path: Path to knowledge base file
    
    Returns:
        List of knowledge items
    """
    if file_path and os.path.exists(file_path):
        try:
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load knowledge base: {e}")
    
    return KNOWLEDGE_BASE
