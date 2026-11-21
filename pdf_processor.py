"""
PDF Processor Module for AI-Powered PDF Assistant.
Sprint 1: PDF Processing using PyMuPDF (fitz).

This module provides modular, beginner-friendly functions to:
1. Open a PDF document safely.
2. Retrieve the total page count.
3. Extract text page-by-page with page number tracking.
4. Handle missing files and invalid PDF formats gracefully.
"""

import os
from typing import Optional, List, Dict, Any
import fitz  # PyMuPDF


def open_pdf(file_path: str) -> Optional[fitz.Document]:
    """
    Safely opens a PDF file using PyMuPDF.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        Optional[fitz.Document]: A fitz Document object if successful, None otherwise.
    """
    # 1. Check if the file exists on the filesystem
    if not os.path.exists(file_path):
        print(f"[Error] File not found: '{file_path}'")
        return None

    # 2. Check if the file has a .pdf extension
    if not file_path.lower().endswith(".pdf"):
        print(f"[Error] Invalid file extension: '{file_path}' is not a PDF file.")
        return None

    try:
        # 3. Attempt to open the document with PyMuPDF
        doc = fitz.open(file_path)
        return doc
    except fitz.FileDataError as e:
        print(f"[Error] Corrupted or invalid PDF file '{file_path}': {e}")
        return None
    except Exception as e:
        print(f"[Error] Unexpected error while opening '{file_path}': {e}")
        return None


def get_page_count(doc: fitz.Document) -> int:
    """
    Returns the total number of pages in an opened PDF document.

    Args:
        doc (fitz.Document): The opened PyMuPDF Document object.

    Returns:
        int: Total page count, or 0 if the document is invalid.
    """
    if doc is None or not isinstance(doc, fitz.Document):
        print("[Error] A valid fitz.Document must be provided to get page count.")
        return 0

    return doc.page_count


def extract_text_page_by_page(doc: fitz.Document) -> List[Dict[str, Any]]:
    """
    Extracts text from each page of an opened PDF document.

    Args:
        doc (fitz.Document): The opened PyMuPDF Document object.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing:
            - 'page_number': 1-based page index.
            - 'text': Extracted text content from the page.
    """
    if doc is None or not isinstance(doc, fitz.Document):
        print("[Error] A valid fitz.Document must be provided to extract text.")
        return []

    pages_data = []

    # Iterate through pages using page indices (0-indexed internally in PyMuPDF)
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)  # Load the specific page
        text = page.get_text()            # Extract plain text from the page

        # Store with 1-based page numbering for user-friendly display
        pages_data.append({
            "page_number": page_index + 1,
            "text": text.strip()
        })

    return pages_data


def process_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Convenience wrapper that opens a PDF, extracts text page-by-page,
    and ensures the document is properly closed.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        List[Dict[str, Any]]: Extracted pages data, or empty list on failure.
    """
    doc = open_pdf(file_path)
    if doc is None:
        return []

    try:
        pages_data = extract_text_page_by_page(doc)
        return pages_data
    finally:
        # Always close the document to release system resources and file locks
        doc.close()
