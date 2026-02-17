"""
RAG (Retrieval-Augmented Generation) module for clinical report generation.
"""
from .retriever import retrieve_knowledge
from .generator import generate_report

__all__ = ['retrieve_knowledge', 'generate_report']
