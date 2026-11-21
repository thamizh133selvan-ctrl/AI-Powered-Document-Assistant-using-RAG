"""
Text Chunker Module for AI-Powered PDF Assistant.
Sprint 2: Text Chunking using LangChain's RecursiveCharacterTextSplitter.

This module provides modular, beginner-friendly functions to:
1. Configure and instantiate a text splitter with a target chunk size and overlap.
2. Split extracted text strings into smaller, coherent chunks for downstream RAG processing.
3. Handle empty or whitespace-only inputs safely.
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_text_splitter(
    chunk_size: int = 500, chunk_overlap: int = 50
) -> RecursiveCharacterTextSplitter:
    """
    Creates and returns a configured RecursiveCharacterTextSplitter instance.

    RecursiveCharacterTextSplitter tries to split on natural boundaries:
    - Double newlines ("\\n\\n") -> Paragraphs
    - Single newlines ("\\n")   -> Sentences/Lines
    - Spaces (" ")             -> Words
    - Empty string ("")        -> Individual characters (fallback)

    Args:
        chunk_size (int): Maximum number of characters per chunk. Defaults to 500.
        chunk_overlap (int): Number of overlapping characters between chunks. Defaults to 50.

    Returns:
        RecursiveCharacterTextSplitter: The configured text splitter instance.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter


def split_text(text: str) -> List[str]:
    """
    Splits input text into smaller chunks using the configured text splitter.

    Args:
        text (str): The raw text string to split.

    Returns:
        List[str]: A list of text chunk strings.
                   Returns an empty list if input is empty, None, or whitespace-only.
    """
    # Defensive check: handle None, non-string, or whitespace-only inputs
    if not text or not isinstance(text, str) or not text.strip():
        return []

    # Create the text splitter with chunk_size=500, chunk_overlap=50
    splitter = create_text_splitter(chunk_size=500, chunk_overlap=50)

    # Perform the split
    chunks = splitter.split_text(text.strip())

    return chunks
