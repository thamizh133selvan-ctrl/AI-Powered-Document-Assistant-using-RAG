"""
Embedding Model Module for AI-Powered PDF Assistant.
Sprint 4 - Step 4: Embedding Generation using Sentence Transformers.

This module provides modular, beginner-friendly functions to:
1. Load and cache the 'all-MiniLM-L6-v2' SentenceTransformer model singleton.
2. Generate a 384-dimensional vector embedding for an individual text string.
3. Generate vector embeddings in batch for a list of text strings.
"""

from typing import Optional, List
from sentence_transformers import SentenceTransformer

# Target embedding model name
MODEL_NAME = "all-MiniLM-L6-v2"

# Module-level variable to cache the loaded model instance (Singleton pattern)
_MODEL_INSTANCE: Optional[SentenceTransformer] = None


def load_embedding_model() -> Optional[SentenceTransformer]:
    """
    Loads and caches the SentenceTransformer embedding model.
    Subsequent calls return the cached model instance rather than reloading.

    Returns:
        Optional[SentenceTransformer]: The loaded SentenceTransformer model,
                                      or None if model loading fails.
    """
    global _MODEL_INSTANCE

    # If the model is already loaded in memory, return the cached instance
    if _MODEL_INSTANCE is not None:
        return _MODEL_INSTANCE

    try:
        print(f"[Info] Loading embedding model '{MODEL_NAME}'...")
        # Attempt loading from local cache first to avoid unnecessary network latency/retries
        try:
            _MODEL_INSTANCE = SentenceTransformer(MODEL_NAME, local_files_only=True)
        except Exception:
            _MODEL_INSTANCE = SentenceTransformer(MODEL_NAME)
        print(f"[Info] Model '{MODEL_NAME}' successfully loaded.")
        return _MODEL_INSTANCE
    except Exception as e:
        print(f"[Error] Failed to load embedding model '{MODEL_NAME}': {e}")
        return None


def generate_embedding(text: str) -> List[float]:
    """
    Generates a 384-dimensional vector embedding for a single text string.

    Args:
        text (str): Input text string to embed.

    Returns:
        List[float]: A list of float numbers representing the text embedding,
                     or an empty list if input is invalid or generation fails.
    """
    # Validate non-empty string input
    if not text or not isinstance(text, str) or not text.strip():
        return []

    # Retrieve the model instance
    model = load_embedding_model()
    if model is None:
        print("[Error] Cannot generate embedding: Model failed to load.")
        return []

    try:
        # Encode text and convert numpy array to a native Python list
        embedding = model.encode(text.strip())
        return embedding.tolist()
    except Exception as e:
        print(f"[Error] Failed to generate embedding for text: {e}")
        return []


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generates 384-dimensional vector embeddings in batch for a list of text strings.
    Preserves the original order of the input texts.

    Args:
        texts (List[str]): List of input text strings to embed.

    Returns:
        List[List[float]]: A list of float lists representing each text's embedding,
                           or an empty list if input is invalid or generation fails.
    """
    # Validate input list
    if not texts or not isinstance(texts, list):
        return []

    # Ensure there is valid text to process
    if not any(isinstance(t, str) and t.strip() for t in texts):
        return []

    # Retrieve the model instance
    model = load_embedding_model()
    if model is None:
        print("[Error] Cannot generate embeddings: Model failed to load.")
        return []

    try:
        # Batch encode all texts, keeping original order
        prepared_texts = [t.strip() if isinstance(t, str) else "" for t in texts]
        embeddings = model.encode(prepared_texts)
        return embeddings.tolist()
    except Exception as e:
        print(f"[Error] Failed to generate batch embeddings: {e}")
        return []
