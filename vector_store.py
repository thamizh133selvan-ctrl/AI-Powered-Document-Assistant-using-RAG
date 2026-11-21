"""
Vector Store Module for AI-Powered PDF Assistant.
Sprint 3 & 4: ChromaDB Connection, Collection Management, Document Storage,
and Sentence Transformer Embedding Integration.

This module provides modular, beginner-friendly functions to:
1. Initialize a persistent ChromaDB client pointing to './chroma_db'.
2. Create or retrieve a collection (defaults to 'pdf_documents').
3. Retrieve an existing collection and handle collection-not-found scenarios gracefully.
4. Add text chunks, metadata, and optional dense embeddings to a collection with unique UUIDs.
5. Retrieve documents and optional embeddings from a collection.
6. Get the total count of records in a collection.
7. Process a PDF file end-to-end:
   extract pages -> split chunks -> generate 384-d embeddings -> store in ChromaDB with metadata.
"""

import os
import re
import uuid
from typing import Optional, List, Dict, Any
import chromadb
from chromadb.config import Settings
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.errors import InvalidCollectionException

from pdf_processor import process_pdf
from text_chunker import split_text
from embedding_model import generate_embeddings

# Default persistent directory for ChromaDB storage
CHROMA_PERSIST_DIR = "./chroma_db"
DEFAULT_COLLECTION_NAME = "pdf_documents"

# Regex pattern to strip institutional/study-material boilerplate headers.
# Matches lines like: "DEPT. OF AIML , JNNCE 4 MACHINE LEARNING STUDY MATERIAL,"
# This header noise dilutes semantic embeddings and hurts retrieval quality.
_BOILERPLATE_HEADER_RE = re.compile(
    r"^(?:DEPT\.\s*OF\s+[A-Z ,]+\s+\d+\s+)?[A-Z ]+STUDY MATERIAL[,.]?\s*",
    re.IGNORECASE | re.MULTILINE
)


def clean_boilerplate_header(text: str) -> str:
    """
    Removes common institutional boilerplate headers from a text chunk before embedding.

    Many scanned study-material PDFs include repeated header lines such as:
        'DEPT. OF AIML , JNNCE 4 MACHINE LEARNING STUDY MATERIAL,'
    When these lines are embedded as part of a chunk they add noise to the dense
    vector representation, lowering cosine similarity to semantically related queries.

    Args:
        text (str): Raw chunk text, possibly beginning with a boilerplate header line.

    Returns:
        str: The text with the leading boilerplate header removed (stripped).
             If the text does not contain a matching header the original text is returned.
    """
    if not text or not isinstance(text, str):
        return text

    cleaned = _BOILERPLATE_HEADER_RE.sub("", text)
    cleaned = cleaned.strip()
    # Fall back to original text if stripping removes everything
    return cleaned if cleaned else text


def delete_documents_by_source(
    collection: Collection,
    source_filename: str
) -> int:
    """
    Deletes all ChromaDB records whose metadata 'source' field matches the given filename.
    Only records belonging to that specific PDF are removed; all other PDFs are preserved.

    Args:
        collection (Collection): An active ChromaDB Collection instance.
        source_filename (str): The exact source filename to match (e.g. 'ML Notes.pdf').

    Returns:
        int: Number of records deleted, or 0 if none found or an error occurs.
    """
    if collection is None:
        print("[Error] A valid ChromaDB Collection must be provided to delete_documents_by_source.")
        return 0

    if not source_filename or not isinstance(source_filename, str) or not source_filename.strip():
        print("[Error] A valid source_filename must be provided to delete_documents_by_source.")
        return 0

    try:
        # Query existing records with matching source metadata (no limit — fetch all)
        existing = collection.get(
            where={"source": source_filename},
            include=[]          # Only need IDs; skip fetching embeddings/documents
        )
        ids_to_delete = existing.get("ids", [])

        if not ids_to_delete:
            print(f"[Info] No existing records found for source '{source_filename}'. Nothing to delete.")
            return 0

        collection.delete(ids=ids_to_delete)
        print(f"[Info] Deleted {len(ids_to_delete)} existing record(s) for source '{source_filename}'.")
        return len(ids_to_delete)

    except Exception as e:
        print(f"[Error] Failed to delete records for source '{source_filename}': {e}")
        return 0


def initialize_chroma(persist_directory: str = CHROMA_PERSIST_DIR) -> Optional[ClientAPI]:
    """
    Initializes and returns a persistent ChromaDB client.

    Args:
        persist_directory (str): File system directory path to store ChromaDB data.
                                 Defaults to './chroma_db'.

    Returns:
        Optional[ClientAPI]: An initialized ChromaDB PersistentClient instance,
                             or None if an initialization error occurs.
    """
    try:
        os.makedirs(persist_directory, exist_ok=True)
        client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        return client
    except Exception as e:
        print(f"[Error] Failed to initialize ChromaDB PersistentClient at '{persist_directory}': {e}")
        return None


def create_collection(
    client: ClientAPI,
    collection_name: str = DEFAULT_COLLECTION_NAME
) -> Optional[Collection]:
    """
    Creates a new ChromaDB collection or retrieves an existing one with the given name.

    Args:
        client (ClientAPI): An active ChromaDB client instance.
        collection_name (str): Name of the collection. Defaults to 'pdf_documents'.

    Returns:
        Optional[Collection]: The ChromaDB Collection instance, or None if creation fails.
    """
    if client is None:
        print("[Error] A valid ChromaDB client must be provided to create or retrieve a collection.")
        return None

    try:
        collection = client.get_or_create_collection(name=collection_name)
        return collection
    except Exception as e:
        print(f"[Error] Failed to create or get collection '{collection_name}': {e}")
        return None


def get_collection(
    client: ClientAPI,
    collection_name: str = DEFAULT_COLLECTION_NAME
) -> Optional[Collection]:
    """
    Retrieves an existing ChromaDB collection by name.

    Args:
        client (ClientAPI): An active ChromaDB client instance.
        collection_name (str): Name of the collection to fetch. Defaults to 'pdf_documents'.

    Returns:
        Optional[Collection]: The ChromaDB Collection instance if found,
                             or None if not found or an error occurs.
    """
    if client is None:
        print("[Error] A valid ChromaDB client must be provided to get a collection.")
        return None

    try:
        collection = client.get_collection(name=collection_name)
        return collection
    except (InvalidCollectionException, ValueError) as e:
        print(f"[Notice] Collection '{collection_name}' not found: {e}")
        return None
    except Exception as e:
        print(f"[Error] Unexpected error while retrieving collection '{collection_name}': {e}")
        return None


def add_documents(
    collection: Collection,
    chunks: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    embeddings: Optional[List[List[float]]] = None
) -> List[str]:
    """
    Adds a list of text chunks with unique UUIDs, metadata, and optional embeddings to ChromaDB.

    Args:
        collection (Collection): An active ChromaDB Collection instance.
        chunks (List[str]): List of text chunk strings to store.
        metadatas (Optional[List[Dict[str, Any]]]): Optional metadata dictionaries corresponding
                                                   to each chunk. Defaults to chunk index.
        embeddings (Optional[List[List[float]]]): Optional list of dense vector embeddings.
                                                 If provided, must match chunk count and dimensions.

    Returns:
        List[str]: A list of generated UUID strings for the added chunks, or [] on failure.
    """
    if collection is None:
        print("[Error] A valid ChromaDB Collection must be provided to add documents.")
        return []

    if not chunks or not isinstance(chunks, list):
        print("[Notice] No text chunks provided to add_documents. Returning empty list.")
        return []

    cleaned_chunks = [c for c in chunks if isinstance(c, str) and c.strip()]
    if not cleaned_chunks:
        print("[Notice] All provided chunks were empty or whitespace-only. Nothing added.")
        return []

    # Validate embeddings if provided
    if embeddings is not None:
        if not isinstance(embeddings, list):
            print("[Error] Provided embeddings must be a list of float lists. Chunks not stored.")
            return []

        # Validate count matches valid chunks count
        if len(embeddings) != len(cleaned_chunks):
            print(f"[Error] Embeddings count ({len(embeddings)}) does not match valid chunks count ({len(cleaned_chunks)}). Chunks not stored.")
            return []

        # Validate consistent embedding dimensions
        first_dim = len(embeddings[0]) if (isinstance(embeddings[0], list) and len(embeddings[0]) > 0) else 0
        if first_dim == 0 or not all(isinstance(e, list) and len(e) == first_dim for e in embeddings):
            print("[Error] Embeddings have invalid or inconsistent dimensions. Chunks not stored.")
            return []

    try:
        # Generate unique UUIDs for each chunk
        ids = [str(uuid.uuid4()) for _ in range(len(cleaned_chunks))]

        # Assign metadata
        if metadatas is None:
            assigned_metadatas = [{"chunk_index": i} for i in range(len(cleaned_chunks))]
        else:
            if len(metadatas) != len(cleaned_chunks):
                print(f"[Warning] Metadatas length ({len(metadatas)}) does not match chunks length ({len(cleaned_chunks)}). Using default metadata.")
                assigned_metadatas = [{"chunk_index": i} for i in range(len(cleaned_chunks))]
            else:
                assigned_metadatas = metadatas

        # Store in ChromaDB collection (with explicit embeddings if supplied)
        if embeddings is not None:
            collection.add(
                ids=ids,
                documents=cleaned_chunks,
                metadatas=assigned_metadatas,
                embeddings=embeddings
            )
        else:
            collection.add(
                ids=ids,
                documents=cleaned_chunks,
                metadatas=assigned_metadatas
            )

        return ids
    except Exception as e:
        print(f"[Error] Failed to add documents to collection '{collection.name}': {e}")
        return []


def get_documents(
    collection: Collection,
    include: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Retrieves stored records (IDs, documents, metadatas, and optional embeddings) from ChromaDB.

    Args:
        collection (Collection): An active ChromaDB Collection instance.
        include (Optional[List[str]]): List of fields to include (e.g. ['embeddings', 'documents', 'metadatas']).

    Returns:
        Dict[str, Any]: Dictionary containing 'ids', 'documents', 'metadatas', and optionally 'embeddings'.
    """
    empty_result = {"ids": [], "documents": [], "metadatas": []}

    if collection is None:
        print("[Error] A valid ChromaDB Collection must be provided to retrieve documents.")
        return empty_result

    try:
        if include:
            results = collection.get(include=include)
        else:
            results = collection.get()

        out = {
            "ids": results.get("ids", []),
            "documents": results.get("documents", []),
            "metadatas": results.get("metadatas", [])
        }

        # Include embeddings if requested and present
        if "embeddings" in results and results["embeddings"] is not None:
            raw_embs = results["embeddings"]
            out["embeddings"] = raw_embs.tolist() if hasattr(raw_embs, "tolist") else raw_embs

        return out
    except Exception as e:
        print(f"[Error] Failed to retrieve documents from collection '{collection.name}': {e}")
        return empty_result


def get_collection_count(collection: Collection) -> int:
    """
    Returns the total count of stored records in a ChromaDB collection.

    Args:
        collection (Collection): An active ChromaDB Collection instance.

    Returns:
        int: Total number of records in the collection, or 0 if invalid or empty.
    """
    if collection is None:
        print("[Error] A valid ChromaDB Collection must be provided to get record count.")
        return 0

    try:
        return collection.count()
    except Exception as e:
        print(f"[Error] Failed to get count for collection '{collection.name}': {e}")
        return 0


def process_and_store_pdf(
    file_path: str,
    collection: Collection
) -> Dict[str, Any]:
    """
    End-to-end integration function:
    1. Extracts text page-by-page using process_pdf().
    2. Splits each page's text into smaller chunks using split_text().
    3. Attaches structured metadata (page_number, source filename, sequential chunk_index).
    4. Generates 384-dimensional vector embeddings using generate_embeddings().
    5. Stores chunks, metadata, and embeddings in ChromaDB using add_documents().

    Args:
        file_path (str): File system path to the target PDF document.
        collection (Collection): An active ChromaDB Collection instance.

    Returns:
        Dict[str, Any]: Summary dictionary containing:
            - 'total_pages': Total pages processed from the PDF
            - 'total_chunks': Total chunks generated
            - 'stored_chunks': Total chunks successfully stored with embeddings
            - 'stored_ids': List of generated document IDs
    """
    empty_summary = {
        "total_pages": 0,
        "total_chunks": 0,
        "stored_chunks": 0,
        "stored_ids": []
    }

    # Guard 1: Validate collection object
    if collection is None:
        print("[Error] A valid ChromaDB Collection must be provided to process_and_store_pdf.")
        return empty_summary

    # Guard 2: Validate file path existence
    if not file_path or not isinstance(file_path, str) or not os.path.exists(file_path):
        print(f"[Error] PDF file not found or invalid path: '{file_path}'")
        return empty_summary

    # Step 1: Extract page-by-page text using existing process_pdf
    pages = process_pdf(file_path)
    if not pages:
        print(f"[Notice] No extractable pages or text found in '{file_path}'.")
        return empty_summary

    source_filename = os.path.basename(file_path)
    all_chunks: List[str] = []
    all_metadatas: List[Dict[str, Any]] = []
    global_chunk_index = 0

    # Step 2: Iterate through each extracted page individually
    for page in pages:
        page_num = page.get("page_number", 1)
        page_text = page.get("text", "")

        if not page_text or not page_text.strip():
            continue

        # Step 3: Split each page's text using existing split_text
        page_chunks = split_text(page_text)

        # Step 4: Build metadata for each chunk, cleaning boilerplate headers before embedding
        for chunk in page_chunks:
            # Strip institutional headers (e.g. "DEPT. OF AIML , JNNCE ... STUDY MATERIAL,")
            # before embedding so the dense vector represents only the semantic content.
            cleaned_chunk = clean_boilerplate_header(chunk)
            all_chunks.append(cleaned_chunk)
            all_metadatas.append({
                "page_number": page_num,
                "source": source_filename,
                "chunk_index": global_chunk_index
            })
            global_chunk_index += 1

    if not all_chunks:
        print(f"[Notice] No text chunks were generated from '{file_path}'.")
        return {
            "total_pages": len(pages),
            "total_chunks": 0,
            "stored_chunks": 0,
            "stored_ids": []
        }

    # Step 5: Generate 384-dimensional embeddings for all chunks
    print(f"[Info] Generating embeddings for {len(all_chunks)} chunks...")
    chunk_embeddings = generate_embeddings(all_chunks)

    # Step 6: Validate embedding generation
    if not chunk_embeddings or len(chunk_embeddings) != len(all_chunks):
        print(f"[Error] Embedding generation failed for '{file_path}'. Text will not be stored without embeddings.")
        return {
            "total_pages": len(pages),
            "total_chunks": len(all_chunks),
            "stored_chunks": 0,
            "stored_ids": []
        }

    # Step 7: Store all chunks along with their embeddings and metadata
    stored_ids = add_documents(
        collection=collection,
        chunks=all_chunks,
        metadatas=all_metadatas,
        embeddings=chunk_embeddings
    )

    return {
        "total_pages": len(pages),
        "total_chunks": len(all_chunks),
        "stored_chunks": len(stored_ids),
        "stored_ids": stored_ids
    }
