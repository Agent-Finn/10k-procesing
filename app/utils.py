"""
Utility functions for the 10-K processing system.
"""
import unicodedata
import re
from typing import List, Dict, Any, Optional
import time


def clean_text(text: str) -> str:
    """Basic text cleaning without AI."""
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def normalize_vector_id(raw_id: str) -> str:
    """Normalize a string to be used as a vector ID."""
    # Replace any non-alphanumeric characters with underscore
    normalized = re.sub(r'[^a-zA-Z0-9]', '_', raw_id)
    # Ensure it's not too long (Pinecone has limits)
    max_length = 100
    if len(normalized) > max_length:
        normalized = normalized[:max_length]
    return normalized


def chunk_text(text, chunk_size=1000, chunk_overlap=200, min_chunk_length=100):
    """
    Chunk text into smaller pieces with overlap.
    
    Args:
        text (str): The text to chunk
        chunk_size (int): The size of each chunk
        chunk_overlap (int): The overlap between chunks
        min_chunk_length (int): Minimum length for a chunk to be included
        
    Returns:
        list: List of text chunks
    """
    if not text or len(text) < min_chunk_length:
        return []
        
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        
        # Don't create chunks that are too small
        if end - start < min_chunk_length:
            break
            
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
        
    return chunks
