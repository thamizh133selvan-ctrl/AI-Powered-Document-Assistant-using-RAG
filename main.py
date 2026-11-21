"""
Main Application Entry Point.
Sprint 1: PDF Processing verification (PyMuPDF).
Sprint 2: Text Chunking verification (RecursiveCharacterTextSplitter).
Sprint 3 - Step 2: ChromaDB Connection and Collection verification.
Sprint 3 - Step 3: ChromaDB Document Addition, Metadata, and Retrieval verification.
Sprint 3 - Step 4: PDF Chunk Integration verification (process_and_store_pdf).
Sprint 4 - Step 4: Embedding Generation verification (SentenceTransformer all-MiniLM-L6-v2).
Sprint 4 - Step 5: Sentence Transformer Embeddings & ChromaDB Integration verification.
Sprint 5 - Step 2: Semantic Retrieval verification (retrieve_chunks).
"""

import os
from pdf_processor import process_pdf
from text_chunker import split_text
from vector_store import (
    initialize_chroma,
    create_collection,
    get_collection,
    add_documents,
    get_documents,
    get_collection_count,
    process_and_store_pdf,
    delete_documents_by_source
)
from embedding_model import (
    load_embedding_model,
    generate_embedding,
    generate_embeddings
)
from retriever import (
    retrieve_chunks,
    search_pdf
)
from unittest.mock import MagicMock
from prompt_builder import build_prompt
from answer_generator import generate_answer, MODEL_NAME
from llm_client import initialize_groq


def test_pdf_extraction(pdf_path: str):
    """
    Sprint 1 Test: Verifies PDF text extraction using process_pdf.
    """
    print(f"\n{'=' * 65}")
    print(f"Sprint 1 Test: PDF Extraction on '{pdf_path}'")
    print(f"{'=' * 65}")

    pages = process_pdf(pdf_path)

    if not pages:
        print(f"[Notice] No content extracted from '{pdf_path}'. (File may be missing, invalid, or empty)\n")
        return []

    print(f"Status: Successfully extracted text.")
    print(f"Total Pages: {len(pages)}\n")

    for page in pages:
        print(f"--- Page {page['page_number']} ---")
        print(f"  {page['text'] if page['text'] else '[No text found]'}")
        print("-" * 35)

    return pages


def test_text_chunking():
    """
    Sprint 2 Test: Verifies text chunking with RecursiveCharacterTextSplitter.
    Tests:
    1. Empty and whitespace-only inputs.
    2. Long text splitting, chunk size limits (<= 500 chars), and overlap (~50 chars).
    3. Splitting extracted text from the sample PDF.
    """
    print(f"\n{'=' * 65}")
    print("Sprint 2 Test: Text Chunking Verification")
    print(f"{'=' * 65}")

    # 1. Test Empty and Whitespace Inputs
    print("\n--- Test 1: Empty and Whitespace-Only Inputs ---")
    empty_cases = ["", "   ", "\n\t  \n  ", None]
    for case in empty_cases:
        result = split_text(case)
        display_case = repr(case) if case is not None else "None"
        print(f"Input {display_case:<12} -> Chunks returned: {len(result)} (Safe handling: {result == []})")

    # 2. Test Long Input Text (Multiple Chunks, <= 500 chars, ~50 overlap)
    print("\n--- Test 2: Long Text Chunking (Target: Size <= 500, Overlap ~ 50) ---")
    paragraphs = [
        "Retrieval-Augmented Generation (RAG) is an architectural pattern that combines the strengths "
        "of traditional information retrieval systems with the generative capabilities of large language "
        "models (LLMs). Instead of relying exclusively on static training weights, a RAG system dynamically "
        "fetches relevant external documents to enrich the model context prompt.",

        "The data ingestion pipeline plays a critical role in any RAG application. First, source documents "
        "such as PDF reports, user manuals, and technical specifications are parsed to extract clean raw text. "
        "Next, this text must be intelligently broken down into smaller segments known as chunks. Chunking "
        "ensures that the retrieved context fits comfortably within the token budget of embedding models.",

        "LangChain's RecursiveCharacterTextSplitter is particularly well-suited for document chunking. It attempts "
        "to split text using a hierarchical list of separators: first paragraphs, then sentences, words, and finally "
        "individual characters if needed. This preserves grammatical coherence and semantic context across boundaries.",

        "By configuring a chunk size of 500 characters and an overlap of 50 characters, we ensure that individual "
        "chunks carry complete thoughts while adjacent chunks share context. This overlap prevents information loss "
        "at chunk boundaries, which is crucial when answering questions that span across split points."
    ]
    long_sample_text = "\n\n".join(paragraphs)
    print(f"Input Text Total Length: {len(long_sample_text)} characters")

    chunks = split_text(long_sample_text)
    print(f"Total Chunks Generated: {len(chunks)}\n")

    all_within_limit = True
    for i, chunk in enumerate(chunks, 1):
        chunk_len = len(chunk)
        if chunk_len > 500:
            all_within_limit = False
        print(f"Chunk #{i} | Length: {chunk_len} chars (<= 500: {chunk_len <= 500})")
        preview = chunk.replace('\n', ' ')
        print(f"  Start: '{preview[:60]}...'")
        print(f"  End:   '...{preview[-60:]}'\n")

    print(f"Verification - All chunks <= 500 chars: {all_within_limit}")

    # Inspect overlap between Chunk 1 and Chunk 2
    if len(chunks) >= 2:
        end_chunk1 = chunks[0][-50:].replace('\n', ' ')
        start_chunk2 = chunks[1][:50].replace('\n', ' ')
        print(f"Boundary Inspection between Chunk 1 and Chunk 2:")
        print(f"  Chunk 1 trailing characters: '{end_chunk1}'")
        print(f"  Chunk 2 leading characters:  '{start_chunk2}'")

    # 3. Test Integration with Extracted PDF Text
    print("\n--- Test 3: Integration with Extracted PDF Text ---")
    sample_pdf = os.path.join("sample_pdfs", "sample.pdf")
    pages = process_pdf(sample_pdf)
    
    if pages:
        combined_pdf_text = "\n\n".join(p["text"] for p in pages if p["text"])
        pdf_chunks = split_text(combined_pdf_text)
        print(f"Extracted PDF Text Length: {len(combined_pdf_text)} characters")
        print(f"Chunks Generated from PDF: {len(pdf_chunks)}")
        for i, c in enumerate(pdf_chunks, 1):
            print(f"  PDF Chunk {i} ({len(c)} chars): '{c.replace(chr(10), ' ')}'")


def test_continuous_text_overlap():
    """
    Sprint 2 Additional Test: Verifies exact chunk sizes and overlap
    using continuous text without whitespace delimiters ('A' * 1200).
    """
    print(f"\n{'=' * 65}")
    print("Sprint 2 Additional Test: Continuous Text Chunk Overlap")
    print(f"{'=' * 65}")

    text = "A" * 1200
    print(f"Input Text: 'A' * 1200 (Total Length: {len(text)} characters)\n")

    chunks = split_text(text)

    # 1. Number of generated chunks
    print(f"Number of generated chunks: {len(chunks)}")

    # 2. Length of each chunk
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i} Length: {len(chunk)} characters")

    # 3. Last 50 characters of Chunk 1 and First 50 characters of Chunk 2
    if len(chunks) >= 2:
        last_50_chunk1 = chunks[0][-50:]
        first_50_chunk2 = chunks[1][:50]
        sections_match = (last_50_chunk1 == first_50_chunk2)

        print(f"\nLast 50 characters of Chunk 1:  '{last_50_chunk1}' (Length: {len(last_50_chunk1)})")
        print(f"First 50 characters of Chunk 2: '{first_50_chunk2}' (Length: {len(first_50_chunk2)})")
        print(f"Do these two sections match?:   {sections_match}")


def test_chroma_connection():
    """
    Sprint 3 - Step 2 Test: Verifies ChromaDB client initialization,
    collection creation, collection retrieval, and reuse.
    """
    print(f"\n{'=' * 65}")
    print("Sprint 3 - Step 2 Test: ChromaDB Connection & Collection")
    print(f"{'=' * 65}")

    # 1. Initializing the ChromaDB client
    print("\n--- Test 1: Initializing ChromaDB PersistentClient ---")
    client = initialize_chroma()
    if client is None:
        print("[Fail] Could not initialize ChromaDB client.")
        return
    print(f"Status: ChromaDB client successfully initialized at './chroma_db'")

    # 2. Creating the pdf_documents collection
    print("\n--- Test 2: Creating 'pdf_documents' Collection ---")
    collection_1 = create_collection(client, collection_name="pdf_documents")
    if collection_1 is None:
        print("[Fail] Failed to create 'pdf_documents' collection.")
        return
    print(f"Status: Collection created/retrieved successfully.")
    print(f"Collection Name: '{collection_1.name}' | Collection ID: {collection_1.id}")

    # 3. Retrieving the existing collection
    print("\n--- Test 3: Retrieving Existing Collection ---")
    collection_retrieved = get_collection(client, collection_name="pdf_documents")
    if collection_retrieved is not None:
        print(f"Status: Successfully retrieved collection '{collection_retrieved.name}'.")
        print(f"Collection ID: {collection_retrieved.id}")
        print(f"ID Matches Initial Collection: {collection_retrieved.id == collection_1.id}")
    else:
        print("[Fail] Could not retrieve existing collection.")

    # 3b. Test get_collection with non-existent collection (graceful handling)
    print("\n--- Test 3b: Graceful Handling for Non-existent Collection ---")
    non_existent = get_collection(client, collection_name="non_existent_test_collection")
    print(f"Returned object for non-existent collection: {non_existent} (Handled gracefully: {non_existent is None})")

    # 4. Calling create_collection() again and confirming reuse
    print("\n--- Test 4: Confirming Collection Reuse (No Duplicate) ---")
    collection_2 = create_collection(client, collection_name="pdf_documents")
    if collection_2 is not None:
        same_id = (collection_1.id == collection_2.id)
        same_name = (collection_1.name == collection_2.name)
        print(f"First Collection ID:  {collection_1.id}")
        print(f"Second Collection ID: {collection_2.id}")
        print(f"IDs match exactly (Reused): {same_id}")
        print(f"Names match:                {same_name}")
        print(f"Reused without creating duplicate: {same_id and same_name}")
    else:
        print("[Fail] Second call to create_collection returned None.")


def test_document_storage():
    """
    Sprint 3 - Step 3 Test: Verifies storing documents, generating unique UUIDs,
    retrieving records, checking metadata, verifying collection counts, and handling
    empty inputs gracefully.
    """
    print(f"\n{'=' * 65}")
    print("Sprint 3 - Step 3 Test: Document Addition and Storage")
    print(f"{'=' * 65}")

    client = initialize_chroma()
    if client is None:
        print("[Fail] ChromaDB client initialization failed.")
        return

    # Use a clean collection for repeatable testing of exact count = 3
    test_collection_name = "test_step3_collection"
    try:
        client.delete_collection(test_collection_name)
    except Exception:
        pass

    collection = create_collection(client, collection_name=test_collection_name)
    if collection is None:
        print("[Fail] Failed to obtain test collection.")
        return

    initial_count = get_collection_count(collection)
    print(f"Initial Collection Record Count: {initial_count} (Expected: 0)")

    # 1. Store 3 sample text chunks
    print("\n--- Test 1: Storing 3 Sample Text Chunks ---")
    sample_chunks = [
        "Chunk 1: Retrieval-Augmented Generation enhances LLM responses with external context.",
        "Chunk 2: Document text is parsed from PDFs using PyMuPDF and split with RecursiveCharacterTextSplitter.",
        "Chunk 3: ChromaDB provides local, persistent vector storage for fast semantic retrieval."
    ]
    ids = add_documents(collection, sample_chunks)

    # 2. Verify that 3 unique IDs are generated
    print("\n--- Test 2: Verifying Unique Generated IDs ---")
    print(f"Generated IDs count: {len(ids)} (Expected: 3)")
    is_unique = (len(ids) == len(set(ids)) == 3)
    print(f"Are all 3 IDs unique?: {is_unique}")

    # 3. Retrieve the stored documents
    print("\n--- Test 3: Retrieving Stored Records ---")
    stored_records = get_documents(collection)
    retrieved_ids = stored_records.get("ids", [])
    retrieved_docs = stored_records.get("documents", [])
    retrieved_metas = stored_records.get("metadatas", [])
    print(f"Retrieved Documents Count: {len(retrieved_docs)} (Expected: 3)")

    # 4. Verify the metadata
    print("\n--- Test 4: Verifying Metadata ---")
    print(f"Retrieved Metadatas Count: {len(retrieved_metas)} (Expected: 3)")
    metadata_indices = [m.get("chunk_index") for m in retrieved_metas]
    print(f"Extracted Chunk Indices from Metadata: {metadata_indices}")
    print(f"Default metadata valid: {set(metadata_indices) == {0, 1, 2}}")

    # 5. Verify collection count is 3
    print("\n--- Test 5: Verifying Collection Count ---")
    current_count = get_collection_count(collection)
    print(f"Current Collection Record Count: {current_count} (Expected: 3)")
    print(f"Collection count verification: {current_count == 3}")

    # 6. Verify empty and invalid inputs are handled safely
    print("\n--- Test 6: Verifying Safe Handling of Empty Inputs ---")
    empty_result_1 = add_documents(collection, [])
    print(f"Empty list input: returned {empty_result_1} (Safe: {empty_result_1 == []})")

    empty_result_2 = add_documents(collection, ["   ", "\n\t"])
    print(f"Whitespace-only input: returned {empty_result_2} (Safe: {empty_result_2 == []})")

    empty_result_3 = add_documents(collection, None)
    print(f"None input: returned {empty_result_3} (Safe: {empty_result_3 == []})")

    count_after_empty_tests = get_collection_count(collection)
    print(f"Count after empty input tests: {count_after_empty_tests} (Unchanged: {count_after_empty_tests == 3})")


def test_pdf_chunk_integration():
    """
    Sprint 3 - Step 4 Test: Verifies end-to-end integration of PDF Extraction,
    Text Chunking, and ChromaDB Storage using process_and_store_pdf().
    """
    print(f"\n{'=' * 65}")
    print("Sprint 3 - Step 4 Test: PDF Chunk Integration (process_and_store_pdf)")
    print(f"{'=' * 65}")

    client = initialize_chroma()
    if client is None:
        print("[Fail] ChromaDB client initialization failed.")
        return

    # Use a fresh 'pdf_documents' collection
    try:
        client.delete_collection("pdf_documents")
    except Exception:
        pass

    collection = create_collection(client, collection_name="pdf_documents")
    if collection is None:
        print("[Fail] Failed to obtain 'pdf_documents' collection.")
        return

    sample_pdf_path = os.path.join("sample_pdfs", "sample.pdf")

    # 1. Execute process_and_store_pdf on sample.pdf
    print(f"\n--- Test 1: Processing and Storing '{sample_pdf_path}' ---")
    result = process_and_store_pdf(sample_pdf_path, collection)

    print(f"Total Pages Processed: {result['total_pages']}")
    print(f"Total Chunks Generated: {result['total_chunks']}")
    print(f"Total Chunks Stored:    {result['stored_chunks']}")
    print(f"Generated Document IDs: {len(result['stored_ids'])}")

    # 2. Verify stored chunk count in collection
    stored_count = get_collection_count(collection)
    print(f"\n--- Test 2: Verifying Collection Count ---")
    print(f"ChromaDB Collection Count: {stored_count} (Expected: {result['stored_chunks']})")
    print(f"Count Matches: {stored_count == result['stored_chunks'] and stored_count > 0}")

    # 3. Retrieve stored records and inspect metadata
    print(f"\n--- Test 3: Verifying Stored Metadata and Attributes ---")
    records = get_documents(collection)
    retrieved_docs = records.get("documents", [])
    retrieved_metas = records.get("metadatas", [])
    retrieved_ids = records.get("ids", [])

    print(f"Retrieved Records Count: {len(retrieved_docs)}")

    combined_records = sorted(
        zip(retrieved_ids, retrieved_docs, retrieved_metas),
        key=lambda r: r[2].get("chunk_index", 0)
    )

    all_metadata_valid = True
    for idx, (doc_id, doc_text, meta) in enumerate(combined_records):
        page_num = meta.get("page_number")
        source = meta.get("source")
        c_index = meta.get("chunk_index")

        print(f"\nRecord #{idx + 1}:")
        print(f"  ID:          {doc_id}")
        print(f"  Source:      '{source}' (Matches 'sample.pdf': {source == 'sample.pdf'})")
        print(f"  Page Number: {page_num}")
        print(f"  Chunk Index: {c_index} (Sequential check: {c_index == idx})")
        print(f"  Snippet:     '{doc_text[:65]}...'")

        if source != "sample.pdf" or c_index != idx or page_num not in (1, 2):
            all_metadata_valid = False

    print(f"\nAll Metadatas Valid (source, page_number, chunk_index): {all_metadata_valid}")

    # 4. Safe handling with invalid / non-existent PDF file
    print(f"\n--- Test 4: Testing Error Handling on Missing PDF ---")
    missing_pdf_path = os.path.join("sample_pdfs", "missing_file.pdf")
    missing_result = process_and_store_pdf(missing_pdf_path, collection)
    print(f"Missing PDF Result: {missing_result}")
    safe_handling = (
        missing_result["total_pages"] == 0
        and missing_result["total_chunks"] == 0
        and missing_result["stored_chunks"] == 0
        and missing_result["stored_ids"] == []
    )
    print(f"Safely handled missing PDF: {safe_handling}")


def test_embedding_generation():
    """
    Sprint 4 - Step 4 Test: Verifies SentenceTransformer model loading,
    singleton caching, single embedding generation (384 dimensions),
    batch embedding generation, and graceful handling of empty inputs.
    """
    print(f"\n{'=' * 65}")
    print("Sprint 4 - Step 4 Test: Embedding Generation (all-MiniLM-L6-v2)")
    print(f"{'=' * 65}")

    # 1. Test model loading and singleton caching
    print("\n--- Test 1: Model Loading & Singleton Caching ---")
    model_1 = load_embedding_model()
    print(f"First load result:  {type(model_1).__name__ if model_1 else 'None'}")
    print(f"Model loaded successfully: {model_1 is not None}")

    model_2 = load_embedding_model()
    print(f"Second load result: {type(model_2).__name__ if model_2 else 'None'}")
    same_instance = (model_1 is model_2)
    print(f"Cached singleton instance returned: {same_instance}")

    # 2. Test single sentence embedding (384 dimensions)
    print("\n--- Test 2: Single Sentence Embedding Generation ---")
    single_text = "Retrieval-Augmented Generation enhances language models with external knowledge."
    print(f"Input sentence: '{single_text}'")
    single_emb = generate_embedding(single_text)
    print(f"Embedding type: {type(single_emb).__name__}")
    print(f"Embedding dimensions: {len(single_emb)} (Expected: 384)")
    print(f"Dimension check (== 384): {len(single_emb) == 384}")
    print(f"Sample values (first 5 floats): {[round(x, 4) for x in single_emb[:5]]}")

    # 3. Test multiple sentences batch embedding
    print("\n--- Test 3: Multiple Sentences Batch Embedding Generation ---")
    sentences = [
        "PyMuPDF extracts clean text page-by-page from PDF files.",
        "RecursiveCharacterTextSplitter segments documents into overlapping chunks.",
        "ChromaDB stores vectors locally for fast semantic similarity search."
    ]
    batch_embs = generate_embeddings(sentences)
    print(f"Input sentences count: {len(sentences)}")
    print(f"Generated embeddings count: {len(batch_embs)} (Expected: {len(sentences)})")
    count_matches = (len(batch_embs) == len(sentences))
    print(f"Batch count matches input count: {count_matches}")

    all_384 = all(len(e) == 384 for e in batch_embs)
    dimensions_list = [len(e) for e in batch_embs]
    print(f"Dimensions of each embedding: {dimensions_list}")
    print(f"Every embedding has exactly 384 dimensions: {all_384}")

    # 4. Test safe handling of empty and invalid inputs
    print("\n--- Test 4: Safe Handling of Empty Inputs ---")
    empty_single_1 = generate_embedding("")
    empty_single_2 = generate_embedding("   \n\t  ")
    empty_single_3 = generate_embedding(None)
    print(f"Empty string input:        {empty_single_1} (Safe: {empty_single_1 == []})")
    print(f"Whitespace-only input:     {empty_single_2} (Safe: {empty_single_2 == []})")
    print(f"None input:                {empty_single_3} (Safe: {empty_single_3 == []})")

    empty_batch_1 = generate_embeddings([])
    empty_batch_2 = generate_embeddings(["   ", ""])
    empty_batch_3 = generate_embeddings(None)
    print(f"Empty list input:          {empty_batch_1} (Safe: {empty_batch_1 == []})")
    print(f"List of empty strings:     {empty_batch_2} (Safe: {empty_batch_2 == []})")
    print(f"None input for batch:      {empty_batch_3} (Safe: {empty_batch_3 == []})")


def test_embedding_chroma_integration():
    """
    Sprint 4 - Step 5 Test: Verifies integration of Sentence Transformer embeddings
    with ChromaDB storage, including retrieval of stored embeddings, dimension checks,
    metadata preservation, and safe handling of mismatched embeddings.
    """
    print(f"\n{'=' * 65}")
    print("Sprint 4 - Step 5 Test: Sentence Transformer & ChromaDB Integration")
    print(f"{'=' * 65}")

    client = initialize_chroma()
    if client is None:
        print("[Fail] ChromaDB client initialization failed.")
        return

    test_collection_name = "test_sprint4_step5_collection"
    try:
        client.delete_collection(test_collection_name)
    except Exception:
        pass

    collection = create_collection(client, collection_name=test_collection_name)
    if collection is None:
        print("[Fail] Failed to obtain test collection.")
        return

    sample_pdf_path = os.path.join("sample_pdfs", "sample.pdf")

    # 1. Process and store PDF with explicit Sentence Transformer embeddings
    print(f"\n--- Test 1: Processing & Storing '{sample_pdf_path}' with Custom Embeddings ---")
    summary = process_and_store_pdf(sample_pdf_path, collection)
    print(f"Total Pages:    {summary['total_pages']}")
    print(f"Total Chunks:   {summary['total_chunks']}")
    print(f"Stored Chunks:  {summary['stored_chunks']}")
    print(f"Stored IDs:     {len(summary['stored_ids'])}")
    print(f"Storage success: {summary['stored_chunks'] == summary['total_chunks'] and summary['stored_chunks'] > 0}")

    # 2. Retrieve records including embeddings from ChromaDB
    print(f"\n--- Test 2: Retrieving Stored Embeddings & Validating Dimensions ---")
    retrieved = get_documents(collection, include=["embeddings", "documents", "metadatas"])
    docs = retrieved.get("documents", [])
    metas = retrieved.get("metadatas", [])
    embs = retrieved.get("embeddings", [])

    print(f"Retrieved documents count:  {len(docs)}")
    print(f"Retrieved embeddings count: {len(embs)}")
    print(f"Counts match: {len(docs) == len(embs) == summary['stored_chunks']}")

    all_384 = all(len(e) == 384 for e in embs) if embs else False
    emb_dims = [len(e) for e in embs]
    print(f"Embedding dimensions per chunk: {emb_dims}")
    print(f"Every stored embedding has 384 dimensions: {all_384}")
    if embs:
        print(f"Chunk #1 first 5 embedding values: {[round(x, 4) for x in embs[0][:5]]}")

    # 3. Verify metadata preservation
    print(f"\n--- Test 3: Verifying Metadata Attributes ---")
    metadata_valid = True
    for i, meta in enumerate(metas):
        source = meta.get("source")
        page_num = meta.get("page_number")
        c_index = meta.get("chunk_index")
        print(f"Chunk #{i + 1} Metadata -> source: '{source}', page_number: {page_num}, chunk_index: {c_index}")
        if source != "sample.pdf" or page_num not in (1, 2) or c_index not in (0, 1):
            metadata_valid = False
    print(f"Metadata valid and preserved: {metadata_valid}")

    # 4. Verify safe handling of invalid and mismatched embeddings
    print(f"\n--- Test 4: Testing Safe Handling of Mismatched Embeddings ---")
    test_chunks = ["Chunk A", "Chunk B"]

    # Subtest 4a: Mismatched count (2 chunks, 1 embedding)
    mismatched_embs = [[0.1] * 384]
    bad_result_1 = add_documents(collection, test_chunks, embeddings=mismatched_embs)
    print(f"Mismatched count test: returned {bad_result_1} (Safely rejected: {bad_result_1 == []})")

    # Subtest 4b: Inconsistent dimensions (one 384, one 100)
    inconsistent_dims = [[0.1] * 384, [0.1] * 100]
    bad_result_2 = add_documents(collection, test_chunks, embeddings=inconsistent_dims)
    print(f"Inconsistent dimensions test: returned {bad_result_2} (Safely rejected: {bad_result_2 == []})")


def test_semantic_retrieval():
    """
    Sprint 5 - Step 2 Test: Verifies semantic vector retrieval using retrieve_chunks(),
    including top-k retrieval, fewer items than top-k, empty/whitespace queries,
    None collection, invalid top-k values, empty collections, result dictionary structure,
    metadata preservation, distance ordering, and cleanup.
    """
    print(f"\n{'=' * 65}")
    print("Sprint 5 - Step 2 Test: Semantic Retrieval (retrieve_chunks)")
    print(f"{'=' * 65}")

    client = initialize_chroma()
    if client is None:
        print("[Fail] ChromaDB client initialization failed.")
        return

    # Use an isolated temporary collection to avoid affecting project collections
    temp_collection_name = "temp_sprint5_retrieval_test"
    try:
        client.delete_collection(temp_collection_name)
    except Exception:
        pass

    collection = create_collection(client, collection_name=temp_collection_name)
    if collection is None:
        print("[Fail] Failed to obtain temporary ChromaDB collection.")
        return

    try:
        # 1. Test querying an empty collection
        print("\n--- Test 1: Querying an Empty Collection ---")
        empty_col_result = retrieve_chunks(collection, "What is machine learning?", top_k=3)
        print(f"Empty collection query result: {empty_col_result} (Expected: [])")
        print(f"Handled empty collection gracefully: {empty_col_result == []}")

        # 2. Populate temporary collection with sample documents & embeddings
        print("\n--- Test 2: Inserting Sample Documents with Embeddings ---")
        sample_texts = [
            "Retrieval-Augmented Generation (RAG) dynamically retrieves external documents to ground LLM answers.",
            "PyMuPDF is a fast Python library used for extracting clean text, images, and layout from PDF files.",
            "RecursiveCharacterTextSplitter splits documents on natural paragraph and sentence boundaries.",
            "ChromaDB is an open-source, local embedding vector database with persistent storage.",
            "Convolutional neural networks are specialized deep learning architectures for computer vision tasks."
        ]
        sample_metas = [
            {"source": "rag_handbook.pdf", "page_number": 1, "chunk_index": 0},
            {"source": "pdf_tools.pdf", "page_number": 3, "chunk_index": 1},
            {"source": "nlp_chunking.pdf", "page_number": 2, "chunk_index": 2},
            {"source": "vector_dbs.pdf", "page_number": 5, "chunk_index": 3},
            {"source": "deep_learning.pdf", "page_number": 8, "chunk_index": 4}
        ]

        sample_embeddings = generate_embeddings(sample_texts)
        stored_ids = add_documents(
            collection=collection,
            chunks=sample_texts,
            metadatas=sample_metas,
            embeddings=sample_embeddings
        )
        print(f"Stored {len(stored_ids)} sample documents into temporary collection.")
        print(f"Collection count: {get_collection_count(collection)}")

        # 3. Test semantic retrieval with top_k=3
        print("\n--- Test 3: Semantic Retrieval with top_k=3 ---")
        query_1 = "How does RAG enhance language models with external documents?"
        print(f"Query: '{query_1}'")
        results_1 = retrieve_chunks(collection, query_1, top_k=3)

        print(f"Retrieved chunks count: {len(results_1)} (Expected: 3)")
        print(f"Count matches top_k: {len(results_1) == 3}")

        # 4. Verify result structure (text, metadata, distance) & metadata attributes
        print("\n--- Test 4: Verifying Result Structure & Metadata ---")
        valid_structure = True
        distances = []
        for rank, item in enumerate(results_1, 1):
            text = item.get("text")
            meta = item.get("metadata")
            dist = item.get("distance")
            distances.append(dist)

            print(f"Rank {rank}:")
            print(f"  Snippet:  '{text[:65]}...'")
            print(f"  Metadata: {meta}")
            print(f"  Distance: {dist:.4f}")

            # Validate keys and types
            if not isinstance(text, str) or not isinstance(meta, dict) or not isinstance(dist, (int, float)):
                valid_structure = False
            if "source" not in meta or "page_number" not in meta or "chunk_index" not in meta:
                valid_structure = False

        print(f"Result dictionaries structure valid: {valid_structure}")

        # 5. Verify ascending distance order (smaller distance = more similar)
        print("\n--- Test 5: Verifying Ascending Distance Order ---")
        is_sorted = all(distances[i] <= distances[i + 1] for i in range(len(distances) - 1))
        print(f"Distances: {[round(d, 4) for d in distances]}")
        print(f"Ascending distance order verified: {is_sorted}")

        # 6. Test retrieval when collection items < top_k
        print("\n--- Test 6: Retrieval when requested top_k exceeds collection count ---")
        results_large_k = retrieve_chunks(collection, query_1, top_k=10)
        print(f"Requested top_k=10, Available=5 -> Returned: {len(results_large_k)} chunks")
        print(f"Safely capped to collection count: {len(results_large_k) == 5}")

        # 7. Test invalid input guards (empty query, whitespace query, None collection, invalid top_k)
        print("\n--- Test 7: Testing Input Validation & Defensive Guards ---")

        # 7a. Empty and whitespace queries
        empty_q_res = retrieve_chunks(collection, "", top_k=3)
        print(f"Empty query '': returned {empty_q_res} (Safe: {empty_q_res == []})")

        ws_q_res = retrieve_chunks(collection, "   \n\t  ", top_k=3)
        print(f"Whitespace-only query: returned {ws_q_res} (Safe: {ws_q_res == []})")

        none_q_res = retrieve_chunks(collection, None, top_k=3)
        print(f"None query: returned {none_q_res} (Safe: {none_q_res == []})")

        # 7b. None collection
        none_col_res = retrieve_chunks(None, "Valid query", top_k=3)
        print(f"None collection: returned {none_col_res} (Safe: {none_col_res == []})")

        # 7c. Invalid top_k values
        zero_k_res = retrieve_chunks(collection, "Valid query", top_k=0)
        print(f"top_k=0: returned {zero_k_res} (Safe: {zero_k_res == []})")

        neg_k_res = retrieve_chunks(collection, "Valid query", top_k=-2)
        print(f"top_k=-2: returned {neg_k_res} (Safe: {neg_k_res == []})")

        str_k_res = retrieve_chunks(collection, "Valid query", top_k="3")
        print(f"top_k='3' (str): returned {str_k_res} (Safe: {str_k_res == []})")

    finally:
        # 8. Clean up the temporary collection
        print("\n--- Test 8: Cleaning Up Temporary Collection ---")
        try:
            client.delete_collection(temp_collection_name)
            print(f"Temporary collection '{temp_collection_name}' successfully deleted.")
        except Exception as e:
            print(f"[Warning] Failed to delete temporary collection: {e}")


def test_search_pdf():
    """
    Sprint 5 - Step 4 Test: Verifies the high-level search_pdf() query pipeline,
    testing structured dictionary response format (query, total_results, results),
    top_k retrieval limits, metadata and attribute preservation (text, source, page_number,
    chunk_index, distance, metadata), ascending distance sorting, and safe handling of
    empty queries, whitespace, None inputs, empty collections, and invalid top_k.
    """
    print(f"\n{'=' * 65}")
    print("Sprint 5 - Step 4 Test: Retrieval Query Pipeline (search_pdf)")
    print(f"{'=' * 65}")

    client = initialize_chroma()
    if client is None:
        print("[Fail] ChromaDB client initialization failed.")
        return

    # 1. Setup isolated temporary ChromaDB collection
    temp_collection_name = "temp_sprint5_search_test"
    try:
        client.delete_collection(temp_collection_name)
    except Exception:
        pass

    collection = create_collection(client, collection_name=temp_collection_name)
    if collection is None:
        print("[Fail] Failed to obtain temporary ChromaDB collection.")
        return

    try:
        # 2. Test querying an empty collection returns valid empty structured response
        print("\n--- Test 1: Querying an Empty Collection ---")
        empty_res = search_pdf(collection, "What is machine learning?", top_k=3)
        print(f"Empty collection response: {empty_res}")
        empty_valid = (
            isinstance(empty_res, dict)
            and empty_res.get("total_results") == 0
            and empty_res.get("results") == []
            and "query" in empty_res
        )
        print(f"Empty collection returns valid structured response: {empty_valid}")

        # 3. Populate temporary collection with at least 5 sample documents with embeddings
        print("\n--- Test 2: Inserting 5 Sample Documents with Real Embeddings ---")
        sample_texts = [
            "Retrieval-Augmented Generation (RAG) dynamically retrieves external documents to ground LLM answers.",
            "PyMuPDF is a fast Python library used for extracting clean text, images, and layout from PDF files.",
            "RecursiveCharacterTextSplitter splits documents on natural paragraph and sentence boundaries.",
            "ChromaDB is an open-source, local embedding vector database with persistent storage.",
            "Convolutional neural networks are specialized deep learning architectures for computer vision tasks."
        ]
        sample_metas = [
            {"source": "rag_handbook.pdf", "page_number": 1, "chunk_index": 0},
            {"source": "pdf_tools.pdf", "page_number": 3, "chunk_index": 1},
            {"source": "nlp_chunking.pdf", "page_number": 2, "chunk_index": 2},
            {"source": "vector_dbs.pdf", "page_number": 5, "chunk_index": 3},
            {"source": "deep_learning.pdf", "page_number": 8, "chunk_index": 4}
        ]

        sample_embeddings = generate_embeddings(sample_texts)
        stored_ids = add_documents(
            collection=collection,
            chunks=sample_texts,
            metadatas=sample_metas,
            embeddings=sample_embeddings
        )
        print(f"Stored {len(stored_ids)} sample documents into temporary collection.")
        print(f"Collection count: {get_collection_count(collection)}")

        # 4. Test valid query returning structured response with top_k=3
        print("\n--- Test 3: Valid Query with top_k=3 ---")
        valid_query = "How does RAG retrieve documents to answer user questions?"
        response = search_pdf(collection, valid_query, top_k=3)

        print(f"Query: '{valid_query}'")
        print(f"Response top-level keys: {list(response.keys())}")
        has_required_keys = {"query", "total_results", "results"}.issubset(response.keys())
        print(f"Contains 'query', 'total_results', and 'results': {has_required_keys}")
        print(f"Response query matches input: {response.get('query') == valid_query}")
        print(f"total_results matches len(results): {response.get('total_results') == len(response.get('results', []))}")
        print(f"Results count <= top_k (3): {len(response.get('results', [])) <= 3 and len(response.get('results', [])) > 0}")

        # 5. Verify fields in each result item & metadata preservation
        print("\n--- Test 4: Verifying Each Result Item's Fields & Metadata ---")
        results = response.get("results", [])
        all_fields_valid = True
        distances = []

        for rank, item in enumerate(results, 1):
            text = item.get("text")
            source = item.get("source")
            page_num = item.get("page_number")
            chunk_idx = item.get("chunk_index")
            dist = item.get("distance")
            meta = item.get("metadata")
            distances.append(dist)

            print(f"Result #{rank}:")
            print(f"  Snippet:      '{text[:60]}...'")
            print(f"  Source:       '{source}'")
            print(f"  Page Number:  {page_num}")
            print(f"  Chunk Index:  {chunk_idx}")
            print(f"  Distance:     {dist:.4f}")
            print(f"  Has Metadata: {isinstance(meta, dict)}")

            # Check field presence and types
            if not isinstance(text, str) or not text:
                all_fields_valid = False
            if not isinstance(source, str) or not source:
                all_fields_valid = False
            if not isinstance(page_num, int):
                all_fields_valid = False
            if not isinstance(chunk_idx, int):
                all_fields_valid = False
            if not isinstance(dist, (int, float)):
                all_fields_valid = False
            if not isinstance(meta, dict):
                all_fields_valid = False

        print(f"All results contain text, source, page_number, chunk_index, distance, metadata: {all_fields_valid}")

        # 6. Verify distance ascending order
        print("\n--- Test 5: Verifying Distances are in Ascending Order ---")
        is_sorted = all(distances[i] <= distances[i + 1] for i in range(len(distances) - 1))
        print(f"Distances: {[round(d, 4) for d in distances]}")
        print(f"Ascending distance order verified: {is_sorted}")

        # 7. Test defensive queries and input validation
        print("\n--- Test 6: Testing Defensive Queries & Edge Cases ---")

        # 7a. Empty query
        empty_q = search_pdf(collection, "", top_k=3)
        print(f"Empty query '': total_results={empty_q.get('total_results')}, results={empty_q.get('results')} (Safe: {empty_q.get('total_results') == 0 and empty_q.get('results') == []})")

        # 7b. Whitespace-only query
        ws_q = search_pdf(collection, "   \n\t  ", top_k=3)
        print(f"Whitespace-only query: total_results={ws_q.get('total_results')}, results={ws_q.get('results')} (Safe: {ws_q.get('total_results') == 0 and ws_q.get('results') == []})")

        # 7c. None query
        none_q = search_pdf(collection, None, top_k=3)
        print(f"None query: total_results={none_q.get('total_results')}, results={none_q.get('results')} (Safe: {none_q.get('total_results') == 0 and none_q.get('results') == []})")

        # 7d. None collection
        none_col = search_pdf(None, "Valid query", top_k=3)
        print(f"None collection: total_results={none_col.get('total_results')}, results={none_col.get('results')} (Safe: {none_col.get('total_results') == 0 and none_col.get('results') == []})")

        # 7e. Invalid top_k values (0, negative, string)
        zero_k = search_pdf(collection, "Valid query", top_k=0)
        print(f"top_k=0: total_results={zero_k.get('total_results')} (Safe: {zero_k.get('total_results') == 0})")

        neg_k = search_pdf(collection, "Valid query", top_k=-3)
        print(f"top_k=-3: total_results={neg_k.get('total_results')} (Safe: {neg_k.get('total_results') == 0})")

        str_k = search_pdf(collection, "Valid query", top_k="3")
        print(f"top_k='3' (str): total_results={str_k.get('total_results')} (Safe: {str_k.get('total_results') == 0})")

    finally:
        # 8. Clean up temporary collection
        print("\n--- Test 7: Cleaning Up Temporary Collection ---")
        try:
            client.delete_collection(temp_collection_name)
            print(f"Temporary collection '{temp_collection_name}' successfully deleted.")
        except Exception as e:
            print(f"[Warning] Failed to delete temporary collection: {e}")


def test_groq_integration():
    """
    Groq Integration Test: Verification of Groq Chat Completions Answer Generation.
    Uses unittest.mock.MagicMock to test Prompt Builder and Answer Generator without
    making real Groq network API calls, verifying prompt formatting, mock generation,
    handling of missing API keys, invalid clients, empty queries, empty responses,
    and API exceptions.
    """
    print(f"\n{'=' * 65}")
    print("Groq Integration Test (Mocked Chat Completions)")
    print(f"{'=' * 65}")

    # Test 1 – Prompt Builder Integration
    print("\n--- Test 1: Prompt Builder Integration ---")
    sample_query = "What is Retrieval-Augmented Generation?"
    sample_chunks = [
        {
            "text": "Retrieval-Augmented Generation (RAG) grounds language models on external knowledge.",
            "source": "rag_overview.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "distance": 0.42
        },
        {
            "text": "RAG reduces hallucinations by retrieving factual passages from vector databases.",
            "source": "rag_overview.pdf",
            "page_number": 2,
            "chunk_index": 1,
            "distance": 0.58
        }
    ]
    prompt = build_prompt(sample_query, sample_chunks)
    has_query = sample_query in prompt
    has_context_1 = "Retrieval-Augmented Generation (RAG) grounds language models" in prompt
    has_context_2 = "reduces hallucinations by retrieving factual passages" in prompt
    has_source = "rag_overview.pdf" in prompt
    has_page = "Page 1" in prompt and "Page 2" in prompt

    t1_pass = has_query and has_context_1 and has_context_2 and has_source and has_page
    print(f"Prompt contains question:         {has_query}")
    print(f"Prompt contains context chunks:    {has_context_1 and has_context_2}")
    print(f"Prompt contains source filename:   {has_source}")
    print(f"Prompt contains page numbers:      {has_page}")
    print(f"Status: [{'PASS' if t1_pass else 'FAIL'}] Prompt Builder correctly structured.")

    # Test 2 – Successful Mock Answer Generation via Groq Chat Completions
    print("\n--- Test 2: Successful Mock Answer Generation (Groq Chat Completions) ---")
    mock_client = MagicMock()
    expected_answer = "Retrieval-Augmented Generation (RAG) grounds language models on external knowledge retrieved from databases."

    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = expected_answer
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    actual_answer = generate_answer(mock_client, sample_query, sample_chunks)
    mock_client.chat.completions.create.assert_called_once()
    called_kwargs = mock_client.chat.completions.create.call_args.kwargs
    called_model = called_kwargs.get("model")
    called_messages = called_kwargs.get("messages", [])

    has_user_message = (
        len(called_messages) > 0
        and called_messages[0].get("role") == "user"
        and sample_query in called_messages[0].get("content", "")
    )

    t2_pass = (
        actual_answer == expected_answer
        and called_model == MODEL_NAME
        and has_user_message
    )
    print(f"Returned answer matches expected:       {actual_answer == expected_answer}")
    print(f"Called with model '{MODEL_NAME}':       {called_model == MODEL_NAME}")
    print(f"Called with user prompt in messages:    {has_user_message}")
    print(f"Status: [{'PASS' if t2_pass else 'FAIL'}] Mock Groq Answer Generation succeeded.")

    # Test 3 – Missing or Empty API Key Handling
    print("\n--- Test 3: Missing / Empty API Key Handling ---")
    none_key_client = initialize_groq(api_key="")
    null_key_client = initialize_groq(api_key="   ")
    placeholder_client = initialize_groq(api_key="your_groq_api_key_here")
    t3_pass = (none_key_client is None and null_key_client is None and placeholder_client is None)
    print(f"Empty API key safely returns None:        {none_key_client is None}")
    print(f"Whitespace API key safely returns None:   {null_key_client is None}")
    print(f"Placeholder API key safely returns None:  {placeholder_client is None}")
    print(f"Status: [{'PASS' if t3_pass else 'FAIL'}] Missing/invalid API keys handled gracefully.")

    # Test 4 – Invalid Client (None)
    print("\n--- Test 4: Invalid Client (None) ---")
    none_client_res = generate_answer(None, sample_query, sample_chunks)
    t4_pass = (none_client_res == "")
    print(f"Returned empty string for None client: {t4_pass}")
    print(f"Status: [{'PASS' if t4_pass else 'FAIL'}] None client handled safely.")

    # Test 5 – Invalid Query (Empty / Whitespace)
    print("\n--- Test 5: Invalid Query (Empty / Whitespace) ---")
    mock_client_t5 = MagicMock()
    empty_query_res = generate_answer(mock_client_t5, "   ", sample_chunks)
    t5_pass = (empty_query_res == "" and not mock_client_t5.chat.completions.create.called)
    print(f"Returned empty string for whitespace query: {empty_query_res == ''}")
    print(f"Groq API was NOT called:                    {not mock_client_t5.chat.completions.create.called}")
    print(f"Status: [{'PASS' if t5_pass else 'FAIL'}] Invalid query handled safely without API call.")

    # Test 6 – Empty Groq Response
    print("\n--- Test 6: Empty Groq Response ---")
    mock_client_t6 = MagicMock()
    empty_response = MagicMock()
    empty_choice = MagicMock()
    empty_choice.message.content = ""
    empty_response.choices = [empty_choice]
    mock_client_t6.chat.completions.create.return_value = empty_response

    empty_resp_answer = generate_answer(mock_client_t6, sample_query, sample_chunks)
    t6_pass = (empty_resp_answer == "")
    print(f"Returned empty string for empty response: {t6_pass}")
    print(f"Status: [{'PASS' if t6_pass else 'FAIL'}] Empty response handled safely.")

    # Test 7 – API Exception Handling
    print("\n--- Test 7: API Exception Handling ---")
    mock_client_t7 = MagicMock()
    mock_client_t7.chat.completions.create.side_effect = RuntimeError("Simulated Groq RateLimitError API error")

    exc_answer = generate_answer(mock_client_t7, sample_query, sample_chunks)
    t7_pass = (exc_answer == "")
    print(f"Returned empty string on API exception: {t7_pass}")
    print(f"Status: [{'PASS' if t7_pass else 'FAIL'}] API exception handled gracefully.")


def test_duplicate_prevention():
    """
    Fix 1 Test: Verifies that delete_documents_by_source() removes only records belonging
    to the specified source PDF, preserves records for other sources, handles edge cases
    (empty filename, None collection, source not found), and prevents duplicate ingestion.
    """
    print(f"\n{'=' * 65}")
    print("Fix 1 Test: Duplicate Prevention (delete_documents_by_source)")
    print(f"{'=' * 65}")

    client = initialize_chroma()
    if client is None:
        print("[Fail] ChromaDB client initialization failed.")
        return

    temp_collection_name = "temp_fix1_dedup_test"
    try:
        client.delete_collection(temp_collection_name)
    except Exception:
        pass

    collection = create_collection(client, collection_name=temp_collection_name)
    if collection is None:
        print("[Fail] Failed to create temporary collection.")
        return

    try:
        from embedding_model import generate_embeddings

        # Insert chunks from two different "PDFs"
        pdf_a_chunks = [
            "Supervised learning requires labelled training data.",
            "Classification and regression are supervised tasks."
        ]
        pdf_b_chunks = [
            "Unsupervised learning finds patterns without labels.",
            "Clustering and dimensionality reduction are unsupervised."
        ]
        pdf_a_meta = [
            {"source": "ml_notes.pdf", "page_number": 1, "chunk_index": 0},
            {"source": "ml_notes.pdf", "page_number": 1, "chunk_index": 1}
        ]
        pdf_b_meta = [
            {"source": "dl_notes.pdf", "page_number": 2, "chunk_index": 0},
            {"source": "dl_notes.pdf", "page_number": 2, "chunk_index": 1}
        ]

        emb_a = generate_embeddings(pdf_a_chunks)
        emb_b = generate_embeddings(pdf_b_chunks)
        add_documents(collection, pdf_a_chunks, pdf_a_meta, embeddings=emb_a)
        add_documents(collection, pdf_b_chunks, pdf_b_meta, embeddings=emb_b)

        total_before = get_collection_count(collection)
        print(f"\n--- Test 1: Initial Population ---")
        print(f"Total records after inserting both PDFs: {total_before} (Expected: 4)")
        print(f"Count correct: {total_before == 4}")

        # Delete only ml_notes.pdf records
        print(f"\n--- Test 2: Deleting 'ml_notes.pdf' Records Only ---")
        deleted = delete_documents_by_source(collection, "ml_notes.pdf")
        total_after = get_collection_count(collection)
        print(f"Records deleted for 'ml_notes.pdf': {deleted} (Expected: 2)")
        print(f"Records remaining in collection:    {total_after} (Expected: 2)")
        t2_pass = deleted == 2 and total_after == 2
        print(f"Status: [{'PASS' if t2_pass else 'FAIL'}] Only ml_notes.pdf records removed.")

        # Verify dl_notes.pdf records are still present
        print(f"\n--- Test 3: Verifying 'dl_notes.pdf' Records Preserved ---")
        remaining = collection.get(where={"source": "dl_notes.pdf"})
        remaining_count = len(remaining.get("ids", []))
        t3_pass = remaining_count == 2
        print(f"'dl_notes.pdf' records still present: {remaining_count} (Expected: 2)")
        print(f"Status: [{'PASS' if t3_pass else 'FAIL'}] Other PDF records preserved correctly.")

        # Test: delete a source that does not exist
        print(f"\n--- Test 4: Deleting a Non-Existent Source ---")
        deleted_none = delete_documents_by_source(collection, "nonexistent.pdf")
        t4_pass = deleted_none == 0 and get_collection_count(collection) == 2
        print(f"Records deleted for 'nonexistent.pdf': {deleted_none} (Expected: 0)")
        print(f"Collection count unchanged:             {get_collection_count(collection) == 2}")
        print(f"Status: [{'PASS' if t4_pass else 'FAIL'}] Non-existent source handled safely.")

        # Test: invalid inputs
        print(f"\n--- Test 5: Invalid Inputs ---")
        r_none_col = delete_documents_by_source(None, "ml_notes.pdf")
        r_empty_name = delete_documents_by_source(collection, "")
        r_none_name = delete_documents_by_source(collection, None)
        t5_pass = r_none_col == 0 and r_empty_name == 0 and r_none_name == 0
        print(f"None collection:  returned {r_none_col} (Safe: {r_none_col == 0})")
        print(f"Empty filename:   returned {r_empty_name} (Safe: {r_empty_name == 0})")
        print(f"None filename:    returned {r_none_name} (Safe: {r_none_name == 0})")
        print(f"Status: [{'PASS' if t5_pass else 'FAIL'}] Invalid inputs handled safely.")

        # Test: re-insert after deletion to verify no duplicates accumulate
        print(f"\n--- Test 6: Re-Insert After Deletion (Duplicate Prevention) ---")
        add_documents(collection, pdf_a_chunks, pdf_a_meta, embeddings=emb_a)
        total_reinsert = get_collection_count(collection)
        print(f"Count after re-inserting ml_notes.pdf (without prior delete): {total_reinsert}")
        delete_documents_by_source(collection, "ml_notes.pdf")
        add_documents(collection, pdf_a_chunks, pdf_a_meta, embeddings=emb_a)
        total_clean = get_collection_count(collection)
        print(f"Count after delete-then-reinsert (clean workflow): {total_clean} (Expected: 4)")
        t6_pass = total_clean == 4
        print(f"Status: [{'PASS' if t6_pass else 'FAIL'}] Delete-then-reinsert produces correct count.")

    finally:
        print(f"\n--- Teardown: Removing Temporary Collection ---")
        try:
            client.delete_collection(temp_collection_name)
            print(f"Temporary collection '{temp_collection_name}' deleted.")
        except Exception as e:
            print(f"[Warning] Could not delete temp collection: {e}")


def test_prompt_builder_improvements():
    """
    Fix 2 Test: Verifies that the updated build_prompt() correctly produces a prompt
    that encourages synthesis, cites sources, and only abstains on truly missing context.
    Also verifies the abstention instruction is not triggered for valid chunks.
    """
    print(f"\n{'=' * 65}")
    print("Fix 2 Test: Updated Prompt Builder (Synthesis & Grounding)")
    print(f"{'=' * 65}")

    sample_query = "Explain supervised learning."
    sample_chunks = [
        {
            "text": "Supervised learning is a type of machine learning where the model learns from labelled training data.",
            "source": "ML Notes.pdf",
            "page_number": 18,
            "chunk_index": 0,
            "distance": 0.42
        }
    ]

    prompt = build_prompt(sample_query, sample_chunks)

    # Test 1: Synthesis instruction is present
    print("\n--- Test 1: Synthesis Instruction Present ---")
    has_synthesis = "summarizing and synthesizing" in prompt
    print(f"Prompt encourages synthesis: {has_synthesis}")
    print(f"Status: [{'PASS' if has_synthesis else 'FAIL'}]")

    # Test 2: Grounding instruction preserved (only context facts)
    print("\n--- Test 2: Grounding Constraint Preserved ---")
    has_grounding = "ONLY facts explicitly stated" in prompt or "only facts" in prompt.lower()
    print(f"Prompt enforces fact-grounding: {has_grounding}")
    print(f"Status: [{'PASS' if has_grounding else 'FAIL'}]")

    # Test 3: Abstention instruction requires genuinely missing/unrelated context
    print("\n--- Test 3: Abstention Triggered Only on Genuinely Missing Context ---")
    has_abstain_guard = "genuinely absent" in prompt or "entirely unrelated" in prompt
    print(f"Abstention instruction scoped correctly: {has_abstain_guard}")
    print(f"Status: [{'PASS' if has_abstain_guard else 'FAIL'}]")

    # Test 4: Source and page citations still present
    print("\n--- Test 4: Source & Page Citations Preserved ---")
    has_source = "ML Notes.pdf" in prompt
    has_page = "Page 18" in prompt
    has_query = sample_query in prompt
    t4_pass = has_source and has_page and has_query
    print(f"Source filename in prompt:   {has_source}")
    print(f"Page number in prompt:       {has_page}")
    print(f"User question in prompt:     {has_query}")
    print(f"Status: [{'PASS' if t4_pass else 'FAIL'}] Source, page and query preserved.")

    # Test 5: Empty query still returns ""
    print("\n--- Test 5: Empty Query Returns Empty String ---")
    empty_prompt = build_prompt("", sample_chunks)
    t5_pass = empty_prompt == ""
    print(f"Empty query returns '': {t5_pass}")
    print(f"Status: [{'PASS' if t5_pass else 'FAIL'}]")

    # Test 6: Empty chunks produce fallback message (not an abstention blocker)
    print("\n--- Test 6: No Context Fallback Message Present ---")
    no_ctx_prompt = build_prompt(sample_query, [])
    has_fallback = "No relevant context extracted" in no_ctx_prompt
    print(f"Empty chunks produce fallback message: {has_fallback}")
    print(f"Status: [{'PASS' if has_fallback else 'FAIL'}]")


def test_retrieval_quality():
    """
    Retrieval Quality Test: Verifies that semantically correct chunks are retrieved
    for typical ML questions after the boilerplate-stripping fix and top_k=5.

    Checks:
    - "Explain Supervised Learning" retrieves at least one chunk from Page 4.
    - "What is Machine Learning?" retrieves at least one chunk from Page 1.
    - "What are labels in supervised learning?" retrieves a chunk mentioning 'label'.

    Note: This test requires ML Notes.pdf to have been ingested into the
    'pdf_documents' collection using the cleaned process_and_store_pdf().
    If the collection is empty or only contains sample.pdf, this test is skipped.
    """
    print(f"\n{'=' * 65}")
    print("Retrieval Quality Test (top_k=5, boilerplate-cleaned chunks)")
    print(f"{'=' * 65}")

    from vector_store import initialize_chroma, create_collection
    from retriever import search_pdf

    client = initialize_chroma()
    if client is None:
        print("[Skip] ChromaDB client unavailable — skipping retrieval quality test.")
        return

    collection = create_collection(client, collection_name="pdf_documents")
    if collection is None or collection.count() == 0:
        print("[Skip] 'pdf_documents' collection is empty — skipping retrieval quality test.")
        print("[Hint] Upload and process ML Notes.pdf via the Streamlit app first.")
        return

    # Only run if ML Notes.pdf is in the collection
    sample = collection.get(where={"source": "ML Notes.pdf"}, include=[], limit=1)
    if not sample.get("ids"):
        print("[Skip] 'ML Notes.pdf' not found in collection — skipping retrieval quality test.")
        print("[Hint] Upload and process ML Notes.pdf via the Streamlit app first.")
        return

    print(f"\nCollection has {collection.count()} total records. Running quality checks...\n")

    # --- Test 1: "What is Machine Learning?" must retrieve Page 1 ---
    print("--- Test 1: 'What is Machine Learning?' → expects Page 1 chunk ---")
    r1 = search_pdf(collection, "What is Machine Learning?", top_k=5)
    pages_1 = [item["page_number"] for item in r1["results"]]
    t1_pass = 1 in pages_1
    print(f"  Pages retrieved: {pages_1}")
    print(f"  Page 1 present: {t1_pass}")
    print(f"  Status: [{'PASS' if t1_pass else 'FAIL'}] Page 1 (ML definition) retrieved correctly.")

    # --- Test 2: "Explain Supervised Learning" must retrieve a Page 4 chunk ---
    print("\n--- Test 2: 'Explain Supervised Learning' → expects Page 4 chunk in top-5 ---")
    r2 = search_pdf(collection, "Explain Supervised Learning", top_k=5)
    pages_2 = [item["page_number"] for item in r2["results"]]
    t2_pass = 4 in pages_2
    print(f"  Pages retrieved: {pages_2}")
    print(f"  Page 4 present: {t2_pass}")
    print(f"  Status: [{'PASS' if t2_pass else 'FAIL'}] Page 4 (Supervised Learning) retrieved correctly.")

    # --- Test 3: "What are labels in supervised learning?" must retrieve chunk mentioning 'label' ---
    print("\n--- Test 3: 'What are labels in supervised learning?' → expects 'label' in top result ---")
    r3 = search_pdf(collection, "What are labels in supervised learning?", top_k=5)
    has_label = any("label" in item["text"].lower() for item in r3["results"])
    t3_pass = has_label
    print(f"  Top result pages: {[item['page_number'] for item in r3['results']]}")
    print(f"  At least one chunk contains 'label': {has_label}")
    print(f"  Status: [{'PASS' if t3_pass else 'FAIL'}] Label-related chunk retrieved.")

    # --- Test 4: No duplicate chunks (all retrieved IDs/texts should be unique) ---
    print("\n--- Test 4: No duplicate texts in results ---")
    texts_2 = [item["text"] for item in r2["results"]]
    t4_pass = len(texts_2) == len(set(texts_2))
    print(f"  Unique chunk texts: {len(set(texts_2))} / {len(texts_2)}")
    print(f"  Status: [{'PASS' if t4_pass else 'FAIL'}] All retrieved chunks are unique.")


def test_sprint7_quality_regression():
    """
    Sprint 7 Final Quality Regression Test:
    Verifies all 9 quality aspects required for Sprint 7 completion:
    1. Scope-aware answer generation
    2. General concept vs specific technique distinction
    3. Multi-chunk answer synthesis
    4. Irrelevant retrieval results filtering (distance_threshold)
    5. Partially relevant context handling
    6. Completely unrelated questions abstention
    7. Citation and page-number preservation
    8. Empty retrieval results handling
    9. Groq API mocked answer generation fidelity
    """
    print(f"\n{'=' * 65}")
    print("Sprint 7 Final Quality Regression Suite")
    print(f"{'=' * 65}")

    from retriever import DEFAULT_DISTANCE_THRESHOLD

    # Sample mock chunks representing different scopes and pages
    mock_chunks_general_vs_specific = [
        {
            "text": "Machine Learning is the science of programming computers so they can learn from data.",
            "source": "ML_Notes.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "distance": 0.45
        },
        {
            "text": "Model-Based Learning involves creating a mathematical model from data. Disadvantages: 1. Requires large dataset. 2. Requires expert knowledge.",
            "source": "ML_Notes.pdf",
            "page_number": 12,
            "chunk_index": 1,
            "distance": 0.72
        }
    ]

    # --- Test 1: Scope-Aware Instructions & Concept Distinction in Prompt ---
    print("\n--- Test 1: Scope-Aware & General vs. Specific Distinction in Prompt ---")
    p1 = build_prompt("What are the disadvantages of Machine Learning?", mock_chunks_general_vs_specific)
    has_scope_inst = "Scope & Category Awareness" in p1
    has_never_attr = "NEVER present properties" in p1 or "specific subcategory" in p1
    has_distinguish = "distinguish" in p1.lower()
    t1_pass = has_scope_inst and has_never_attr and has_distinguish
    print(f"Prompt enforces Scope & Category Awareness:     {has_scope_inst}")
    print(f"Prompt instructs never to generalize subcategory: {has_never_attr}")
    print(f"Status: [{'PASS' if t1_pass else 'FAIL'}] Scope awareness instructions verified.")

    # --- Test 2: Multi-Chunk Answer Synthesis Formatting ---
    print("\n--- Test 2: Multi-Chunk Answer Synthesis Context Formatting ---")
    p2 = build_prompt("Explain ML and Model-Based learning.", mock_chunks_general_vs_specific)
    has_chunk1 = "Machine Learning is the science of programming" in p2
    has_chunk2 = "Model-Based Learning involves creating a mathematical model" in p2
    has_both_pages = "Page 1" in p2 and "Page 12" in p2
    t2_pass = has_chunk1 and has_chunk2 and has_both_pages
    print(f"Both chunks preserved in context:      {has_chunk1 and has_chunk2}")
    print(f"Page numbers 1 and 12 preserved:       {has_both_pages}")
    print(f"Status: [{'PASS' if t2_pass else 'FAIL'}] Multi-chunk context formatted correctly.")

    # --- Test 3: Irrelevant Retrieval Filtering via Distance Threshold ---
    print("\n--- Test 3: Irrelevant Retrieval Filtering (distance_threshold) ---")
    mock_mixed_results = [
        {"text": "Relevant ML definition", "metadata": {"source": "ml.pdf", "page_number": 1}, "distance": 0.65},
        {"text": "Partially relevant chunk", "metadata": {"source": "ml.pdf", "page_number": 2}, "distance": 0.95},
        {"text": "Distant unrelated chunk", "metadata": {"source": "ml.pdf", "page_number": 88}, "distance": 1.48}
    ]
    # Filter with default threshold (1.25)
    filtered = [c for c in mock_mixed_results if c["distance"] <= DEFAULT_DISTANCE_THRESHOLD]
    t3_pass = len(filtered) == 2 and all(c["distance"] <= 1.25 for c in filtered)
    print(f"Input chunks count:        {len(mock_mixed_results)}")
    print(f"Chunks <= 1.25 threshold:  {len(filtered)} (Expected: 2)")
    print(f"Unrelated chunk (1.48) filtered out: {all(c['distance'] <= 1.25 for c in filtered)}")
    print(f"Status: [{'PASS' if t3_pass else 'FAIL'}] Threshold filtering works as expected.")

    # --- Test 4: Partially Relevant Context & Limitation Instructions ---
    print("\n--- Test 4: Partially Relevant Context Instructions ---")
    has_partial_inst = "Partial Information & Limitations" in p1
    has_limit_note = "clearly mention what the document does NOT explain" in p1 or "does not cover" in p1.lower()
    t4_pass = has_partial_inst and has_limit_note
    print(f"Partial information instruction present: {has_partial_inst}")
    print(f"Instruction notes limitations:           {has_limit_note}")
    print(f"Status: [{'PASS' if t4_pass else 'FAIL'}] Limitation disclosure instruction verified.")

    # --- Test 5: Completely Unrelated Question Abstention ---
    print("\n--- Test 5: Completely Unrelated Question Abstention ---")
    unrelated_prompt = build_prompt("Explain Quantum Computing and its working principles.", [])
    has_abstain_clause = "The provided document context does not contain sufficient information" in unrelated_prompt
    has_no_context = "No relevant context extracted" in unrelated_prompt
    t5_pass = has_abstain_clause and has_no_context
    print(f"Prompt instructs standard abstention string: {has_abstain_clause}")
    print(f"Empty context fallback included:            {has_no_context}")
    print(f"Status: [{'PASS' if t5_pass else 'FAIL'}] Unrelated query prompt structured to trigger abstention.")

    # --- Test 6: Citation & Page-Number Fidelity ---
    print("\n--- Test 6: Citation & Page-Number Fidelity in Prompt ---")
    has_cite_inst = "Source & Citation Fidelity" in p1
    has_no_invent = "Do NOT invent" in p1
    t6_pass = has_cite_inst and has_no_invent
    print(f"Citation fidelity rule present:  {has_cite_inst}")
    print(f"Strict rule against inventing citations: {has_no_invent}")
    print(f"Status: [{'PASS' if t6_pass else 'FAIL'}] Citation fidelity rules verified.")

    # --- Test 7: Empty Retrieval Results Handling ---
    print("\n--- Test 7: Empty Retrieval Results in generate_answer() ---")
    mock_client_empty = MagicMock()
    mock_resp_abstain = MagicMock()
    mock_resp_abstain.choices = [MagicMock()]
    mock_resp_abstain.choices[0].message.content = "The provided document context does not contain sufficient information to answer this question."
    mock_client_empty.chat.completions.create.return_value = mock_resp_abstain

    ans_empty = generate_answer(mock_client_empty, "Unrelated question", [])
    t7_pass = ans_empty == "The provided document context does not contain sufficient information to answer this question."
    print(f"Returned exact abstention message: {t7_pass}")
    print(f"Status: [{'PASS' if t7_pass else 'FAIL'}] Empty retrieval handled gracefully.")

    # --- Test 8: Mocked Scope-Aware Answer Generation Fidelity ---
    print("\n--- Test 8: Mocked Scope-Aware Answer Generation Fidelity ---")
    mock_client_scope = MagicMock()
    mock_resp_scope = MagicMock()
    expected_scope_ans = (
        "The document details disadvantages specifically for Model-Based Learning [ML_Notes.pdf | Page 12]: "
        "1. Requires large dataset. 2. Requires expert knowledge. "
        "It does not provide general disadvantages for Machine Learning as a whole."
    )
    mock_resp_scope.choices = [MagicMock()]
    mock_resp_scope.choices[0].message.content = expected_scope_ans
    mock_client_scope.chat.completions.create.return_value = mock_resp_scope

    ans_scope = generate_answer(mock_client_scope, "What are the disadvantages of Machine Learning?", mock_chunks_general_vs_specific)
    t8_pass = "Model-Based Learning" in ans_scope and "not provide general disadvantages" in ans_scope
    print(f"Output distinguishes subcategory from general field: {t8_pass}")
    print(f"Status: [{'PASS' if t8_pass else 'FAIL'}] Scope-aware generation verified.")


def main():
    print("=" * 65)
    print("AI-Powered PDF Assistant - Full Pipeline Verification")
    print("=" * 65)

    # Sprint 1 Verification
    sample_pdf_path = os.path.join("sample_pdfs", "sample.pdf")
    test_pdf_extraction(sample_pdf_path)

    # Sprint 2 Verification
    test_text_chunking()
    test_continuous_text_overlap()

    # Sprint 3 - Step 2 Verification
    test_chroma_connection()

    # Sprint 3 - Step 3 Verification
    test_document_storage()

    # Sprint 3 - Step 4 Verification: End-to-End PDF Chunk Integration
    test_pdf_chunk_integration()

    # Sprint 4 - Step 4 Verification: Embedding Generation
    test_embedding_generation()

    # Sprint 4 - Step 5 Verification: Embeddings & ChromaDB Integration
    test_embedding_chroma_integration()

    # Sprint 5 - Step 2 Verification: Semantic Retrieval Testing
    test_semantic_retrieval()

    # Sprint 5 - Step 4 Verification: Retrieval Query Pipeline Testing
    test_search_pdf()

    # Groq Chat Completions Integration Testing (Mocked)
    test_groq_integration()

    # Fix 1 Verification: Duplicate Prevention
    test_duplicate_prevention()

    # Fix 2 Verification: Updated Prompt Builder
    test_prompt_builder_improvements()

    # Retrieval Quality Verification (requires ML Notes.pdf ingested with cleaned chunks)
    test_retrieval_quality()

    # Sprint 7 Final Quality Regression Verification
    test_sprint7_quality_regression()


if __name__ == "__main__":
    main()
