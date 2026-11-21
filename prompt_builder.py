"""
Prompt Builder Module for AI-Powered PDF Assistant.
Sprint 7: Grounded, Scope-Aware RAG Prompt Construction for Groq API.

This module provides functions to construct grounded, scope-aware prompts
for LLMs combining user queries and semantically retrieved PDF text chunks.
"""

from typing import List, Dict, Any, Optional


def build_prompt(query: str, retrieved_chunks: Optional[List[Dict[str, Any]]]) -> str:
    """
    Constructs a grounded, structured prompt for the LLM combining user query and retrieved PDF chunks.

    Enforces:
    1. Strict grounding on provided context only.
    2. Scope & category awareness: distinguishing overarching concepts from specific subcategories.
    3. Partial information disclosure: reporting what is known and noting what is not covered.
    4. Multi-chunk synthesis without distortion.
    5. Clean citation of document names and page numbers.
    6. Safe abstention on missing or unrelated context.

    Args:
        query (str): The user's question or search query.
        retrieved_chunks (Optional[List[Dict[str, Any]]]): A list of chunk dictionaries
            retrieved from the vector store (e.g. from retrieve_chunks or search_pdf).

    Returns:
        str: A formatted prompt string ready for LLM generation.
             Returns an empty string "" if the query is invalid, empty, or whitespace-only.
    """
    # 1. Validate the query: empty, whitespace-only or non-string queries return empty string
    if not query or not isinstance(query, str) or not query.strip():
        return ""

    # 2. Validate retrieved_chunks list safely
    if not isinstance(retrieved_chunks, list):
        retrieved_chunks = []

    # 3. Extract and format valid context blocks from chunks
    context_blocks: List[str] = []

    for idx, chunk in enumerate(retrieved_chunks, 1):
        if not isinstance(chunk, dict):
            continue

        # Extract text content and ignore chunks with missing or empty text
        text = chunk.get("text")
        if not text or not isinstance(text, str) or not text.strip():
            continue

        # Extract source and page metadata with fallbacks for both retrieve_chunks and search_pdf formats
        raw_meta = chunk.get("metadata")
        meta = raw_meta if isinstance(raw_meta, dict) else {}

        source = chunk.get("source") or meta.get("source") or "Unknown Source"
        page_num = chunk.get("page_number")
        if page_num is None:
            page_num = meta.get("page_number")

        page_str = f"Page {page_num}" if page_num is not None else "Page N/A"

        # Format context item with clear document and page attribution
        context_blocks.append(f"[{source} | {page_str}]\n{text.strip()}")

    # 4. Handle cases where no valid text chunks are available
    if context_blocks:
        context_section = "\n\n".join(context_blocks)
    else:
        context_section = "No relevant context extracted from PDF documents."

    # 5. Assemble structured prompt enforcing grounded, scope-aware question-answering
    prompt = (
        "You are an AI-powered PDF Assistant specializing in precise, factual question-answering based strictly on extracted document content.\n\n"
        "Instructions:\n"
        "1. Grounding & Factual Accuracy: Answer the user's question by summarizing and synthesizing ONLY facts explicitly stated or directly supported by the provided PDF context below. Do not introduce outside knowledge, unstated assumptions, or speculative extrapolations.\n"
        "2. Scope & Category Awareness: Carefully distinguish overarching general concepts (e.g., Machine Learning as a whole) from specific techniques, algorithms, or subcategories (e.g., Model-Based Learning, Instance-Based Learning, Supervised Learning, Find-S).\n"
        "   - NEVER present properties, advantages, or disadvantages of a specific subcategory as universal properties of the broader field.\n"
        "   - If the user asks about the general concept (e.g., advantages or disadvantages of Machine Learning), but the context only provides details for a specific subcategory (e.g., Model-Based Learning), explicitly clarify that distinction: state that the document details advantages/disadvantages specifically for that subcategory, and note that general advantages/disadvantages for the entire field are not provided in the text.\n"
        "3. Partial Information & Limitations: If the provided context covers only part of the question, answer only what is supported by the context and clearly mention what the document does NOT explain or cover. Never invent details to fill gaps.\n"
        "4. Multi-Chunk Synthesis: When combining information across multiple chunks and pages, preserve the original meaning, context, and scope of each chunk without distorting or conflating them.\n"
        "5. Abstention: If the provided context is empty, missing, or entirely unrelated to the question, respond ONLY with:\n"
        "   \"The provided document context does not contain sufficient information to answer this question.\"\n"
        "6. Source & Citation Fidelity: Cite the exact source document name and page number (e.g., [Document.pdf | Page X]) for every fact, claim, or section in your answer. Do NOT invent, assume, or cite page numbers or documents not present in the context.\n\n"
        "--- Context ---\n"
        f"{context_section}\n"
        "----------------\n\n"
        f"User Question: {query.strip()}\n\n"
        "Answer:"
    )

    return prompt
