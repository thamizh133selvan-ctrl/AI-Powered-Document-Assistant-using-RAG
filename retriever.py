"""
Retriever Module for AI-Powered PDF Assistant.
Sprint 5 & Sprint 7: Semantic Retrieval with ChromaDB and SentenceTransformer Embeddings.

This module provides modular, beginner-friendly functions to:
1. Validate queries and search parameters.
2. Compute dense query embeddings using the existing SentenceTransformer model.
3. Perform vector similarity search against a ChromaDB collection using collection.query().
4. Filter by configurable distance relevance threshold when provided.
5. Deduplicate identical text passages to maintain clean source citations.
6. Parse and return matching chunks along with their metadata and distance scores.
"""

from typing import List, Dict, Any, Optional
from chromadb.api.models.Collection import Collection
from embedding_model import generate_embedding

# Default relevance distance threshold for all-MiniLM-L6-v2 in ChromaDB.
# Distances <= 1.25 represent semantically related domain passages,
# while completely unrelated queries (e.g. quantum computing, recipes) typically score > 1.35.
DEFAULT_DISTANCE_THRESHOLD: float = 1.25


def retrieve_chunks(
    collection: Optional[Collection],
    query: str,
    top_k: int = 3,
    distance_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Performs semantic vector search over a ChromaDB collection to retrieve the most relevant text chunks.

    Args:
        collection (Optional[Collection]): The target ChromaDB Collection to search.
        query (str): The search query text string.
        top_k (int): Number of top matching chunks to retrieve. Defaults to 3.
        distance_threshold (Optional[float]): Optional maximum distance threshold.
            Chunks with distance > distance_threshold are filtered out.
            Defaults to None (no filtering, returns raw top_k).

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing:
            - 'text': Extracted chunk text (str)
            - 'metadata': Associated metadata dict (e.g., page_number, source, chunk_index)
            - 'distance': Semantic distance score (float; smaller distance means higher similarity)
            Returns an empty list [] on invalid input, empty results, or query errors.
    """
    # 1. Input Validation: Collection must not be None
    if collection is None:
        print("[Error] A valid ChromaDB Collection must be provided to retrieve_chunks.")
        return []

    # 2. Input Validation: Query must be a non-empty string
    if not query or not isinstance(query, str) or not query.strip():
        print("[Notice] Query must be a non-empty string. Returning empty list.")
        return []

    # 3. Input Validation: top_k must be a positive integer
    if not isinstance(top_k, int) or top_k <= 0:
        print(f"[Error] top_k must be a positive integer, got: {top_k}. Returning empty list.")
        return []

    # 4. Generate dense embedding for the query using the same model as our stored chunks
    query_embedding = generate_embedding(query.strip())
    if not query_embedding or len(query_embedding) == 0:
        print("[Error] Failed to generate query embedding. Returning empty list.")
        return []

    # 5. Query ChromaDB using explicit query_embeddings (never query_texts)
    try:
        count = collection.count()
        if count == 0:
            print(f"[Notice] Collection '{collection.name}' is empty. Returning empty list.")
            return []

        # Ensure n_results does not exceed total available items in the collection
        effective_k = min(top_k, count)

        raw_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_k,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        print(f"[Error] Exception occurred during ChromaDB query: {e}")
        return []

    # 6. Parse, filter by threshold if specified, and format results
    retrieved_items: List[Dict[str, Any]] = []

    try:
        documents_list = raw_results.get("documents", [[]])[0]
        metadatas_list = raw_results.get("metadatas", [[]])[0]
        distances_list = raw_results.get("distances", [[]])[0]

        for idx in range(len(documents_list)):
            doc_text = documents_list[idx]
            meta = metadatas_list[idx] if idx < len(metadatas_list) else {}
            dist = distances_list[idx] if idx < len(distances_list) else 0.0

            # Filter out chunks exceeding the distance threshold if one is specified
            if distance_threshold is not None and dist > distance_threshold:
                continue

            retrieved_items.append({
                "text": doc_text,
                "metadata": meta,
                "distance": dist
            })

    except Exception as e:
        print(f"[Error] Unexpected format in ChromaDB query response: {e}")
        return []

    return retrieved_items


def search_pdf(
    collection: Optional[Collection],
    query: str,
    top_k: int = 3,
    distance_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    High-level query pipeline to search stored PDF chunks and return a structured response.

    This function reuses retrieve_chunks() for semantic vector retrieval, extracting
    and organizing chunk details into a standardized dictionary format, while preventing
    duplicate texts from appearing in the results.

    Args:
        collection (Optional[Collection]): The ChromaDB collection containing indexed document chunks.
        query (str): The search question or keyword query.
        top_k (int): Maximum number of top matching chunks to retrieve. Defaults to 3.
        distance_threshold (Optional[float]): Optional maximum distance threshold.
            Defaults to None (returns all top_k without distance cut-off).

    Returns:
        Dict[str, Any]: A structured dictionary containing:
            - 'query' (str): The search query processed.
            - 'total_results' (int): Count of retrieved results.
            - 'results' (List[Dict[str, Any]]): List of formatted result items.
    """
    # 1. Initialize a safe, default response structure for invalid queries or no results
    safe_query_str = query if isinstance(query, str) else ""
    default_response: Dict[str, Any] = {
        "query": safe_query_str,
        "total_results": 0,
        "results": []
    }

    # 2. Early exit with valid structured response if query is empty, whitespace, or invalid
    if not isinstance(query, str) or not query.strip():
        return default_response

    # 3. Retrieve chunks with optional distance threshold
    raw_chunks = retrieve_chunks(
        collection=collection,
        query=query,
        top_k=top_k,
        distance_threshold=distance_threshold
    )

    # If no chunks are found or an error occurred, return empty valid structure
    if not raw_chunks:
        return default_response

    # 4. Format each retrieved chunk, deduplicating identical texts
    formatted_results: List[Dict[str, Any]] = []
    seen_texts = set()

    for item in raw_chunks:
        chunk_text = item.get("text", "")
        # Prevent identical duplicate chunks from cluttering citations
        if chunk_text in seen_texts:
            continue
        seen_texts.add(chunk_text)

        meta = item.get("metadata")
        if not isinstance(meta, dict):
            meta = {}

        source = meta.get("source", "unknown")
        page_number = meta.get("page_number", None)
        chunk_index = meta.get("chunk_index", None)
        distance = item.get("distance", 0.0)

        formatted_results.append({
            "text": chunk_text,
            "source": source,
            "page_number": page_number,
            "chunk_index": chunk_index,
            "distance": distance,
            "metadata": meta
        })

    # 5. Return complete structured response
    return {
        "query": query,
        "total_results": len(formatted_results),
        "results": formatted_results
    }
