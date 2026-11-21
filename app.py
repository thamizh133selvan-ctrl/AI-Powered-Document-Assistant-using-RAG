"""
Streamlit Web Application for AI-Powered PDF Assistant.
Sprint 7: Production-Grade RAG Chat Interface with Groq API (openai/gpt-oss-120b).

Features:
1. Modern, polished responsive UI with custom styling and dark/light mode compatibility.
2. Drag-and-drop PDF ingestion with real-time indexing status and persistent metrics.
3. Interactive Q&A chat interface with streaming-like visual flow and sample prompt chips.
4. Source citation cards displaying exact page numbers, relevance distance scores, and snippets.
5. Sidebar controls for chat history, vector collection health, and live engine status.
"""

import os
import tempfile
import streamlit as st
from vector_store import (
    initialize_chroma,
    create_collection,
    process_and_store_pdf,
    delete_documents_by_source
)
from retriever import search_pdf, DEFAULT_DISTANCE_THRESHOLD
from llm_client import initialize_groq
from answer_generator import generate_answer, MODEL_NAME


# -----------------------------------------------------------------------------
# Cached Resource Initializers
# -----------------------------------------------------------------------------

def get_vector_collection():
    """
    Initializes ChromaDB client and retrieves the target 'pdf_documents' collection.
    Caches the collection in Streamlit session state for performance across reruns.
    """
    if "chroma_collection" not in st.session_state or st.session_state["chroma_collection"] is None:
        client = initialize_chroma()
        if client is not None:
            collection = create_collection(client, collection_name="pdf_documents")
            st.session_state["chroma_collection"] = collection
        else:
            st.session_state["chroma_collection"] = None

    return st.session_state["chroma_collection"]


def get_groq_client():
    """
    Initializes and caches the Groq client in session state.
    Loads GROQ_API_KEY from .env safely without exposing credentials.
    """
    if "groq_client" not in st.session_state or st.session_state["groq_client"] is None:
        st.session_state["groq_client"] = initialize_groq()

    return st.session_state["groq_client"]


# -----------------------------------------------------------------------------
# Custom Styling (CSS)
# -----------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* Global Container & Typography */
.main-header {
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(128, 128, 128, 0.2);
}

.title-container {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.2rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    background-color: rgba(16, 185, 129, 0.15);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: #10B981;
    display: inline-block;
}

/* Metric Cards */
.metric-card {
    border-radius: 0.75rem;
    padding: 1rem;
    background: rgba(128, 128, 128, 0.06);
    border: 1px solid rgba(128, 128, 128, 0.15);
    text-align: center;
    transition: all 0.2s ease-in-out;
}
.metric-card:hover {
    border-color: rgba(79, 70, 229, 0.4);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #4F46E5;
    margin-bottom: 0.1rem;
}
.metric-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    opacity: 0.75;
}

/* Citation Badges */
.citation-card {
    border-radius: 0.6rem;
    padding: 0.85rem;
    margin-bottom: 0.75rem;
    background: rgba(128, 128, 128, 0.05);
    border-left: 3px solid #4F46E5;
}
.citation-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
    font-size: 0.82rem;
    font-weight: 600;
}
.badge-pill {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 0.375rem;
    font-size: 0.75rem;
    font-weight: 600;
    background: rgba(79, 70, 229, 0.12);
    color: #4F46E5;
}
.dist-pill {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 0.375rem;
    font-size: 0.72rem;
    opacity: 0.75;
    background: rgba(128, 128, 128, 0.15);
}
.citation-snippet {
    font-size: 0.85rem;
    line-height: 1.45;
    opacity: 0.9;
    font-style: italic;
}

/* Welcome Card */
.welcome-card {
    border-radius: 0.85rem;
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
    background: rgba(79, 70, 229, 0.04);
    border: 1px dashed rgba(79, 70, 229, 0.3);
}
</style>
"""


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------

def main():
    # 1. Page Configuration
    st.set_page_config(
        page_title="AI PDF Assistant | Groq RAG",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inject Custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # 2. Sidebar Configuration & Architecture Controls
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <span style="font-size: 1.6rem;">⚡</span>
                <span style="font-size: 1.2rem; font-weight: 700;">System Specs</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        groq_client_check = get_groq_client()
        collection = get_vector_collection()
        doc_count = collection.count() if collection else 0

        # Provider & Model details
        st.markdown(
            f"""
            <div style="padding: 0.75rem; border-radius: 0.5rem; background: rgba(128, 128, 128, 0.08); margin-bottom: 1rem;">
                <div style="font-size: 0.75rem; text-transform: uppercase; opacity: 0.7; font-weight: 600;">Inference Engine</div>
                <div style="font-weight: 700; color: #F97316;">Groq Cloud API</div>
                <div style="font-size: 0.8rem; font-family: monospace; margin-top: 0.2rem;">{MODEL_NAME}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("**Vector Store:** ChromaDB (`./chroma_db`)")
        st.markdown(f"**Embeddings:** `all-MiniLM-L6-v2` (384-d)")
        st.markdown(f"**Total Indexed Chunks:** `{doc_count}`")

        st.divider()

        st.subheader("🎯 Retrieval Parameters")
        top_k = st.slider("Top Chunks (k)", min_value=1, max_value=8, value=5, step=1, help="Number of semantic chunks retrieved per question.")
        dist_threshold = st.slider(
            "Max Distance Cutoff",
            min_value=0.5,
            max_value=1.6,
            value=float(DEFAULT_DISTANCE_THRESHOLD),
            step=0.05,
            help="Threshold cutoff: lower distance means higher similarity. Chunks with distance above this value are rejected as unrelated."
        )

        st.divider()

        # Engine Health Status
        st.markdown("**Engine Health:**")
        if groq_client_check:
            st.markdown("🟢 `Groq Client`: **Ready**")
        else:
            st.markdown("🔴 `Groq Client`: **API Key Missing**")

        if collection is not None:
            st.markdown("🟢 `ChromaDB`: **Connected**")
        else:
            st.markdown("🔴 `ChromaDB`: **Disconnected**")

        st.divider()

        # Chat Control Buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state["chat_history"] = []
                st.rerun()
        with col_btn2:
            if st.button("🔄 Reload App", use_container_width=True):
                st.rerun()

    # 3. Main Header Area
    col_header, col_status = st.columns([0.72, 0.28])
    with col_header:
        st.markdown(
            """
            <div class="title-container">
                <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800;">📄 AI-Powered PDF Assistant</h1>
            </div>
            <p style="margin-top: 0.35rem; font-size: 0.98rem; opacity: 0.85;">
                Ingest PDF documents, generate local 384-d vector embeddings, and ask grounded questions powered by <b>Groq (openai/gpt-oss-120b)</b> with exact page-level citations.
            </p>
            """,
            unsafe_allow_html=True
        )
    with col_status:
        st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)
        if groq_client_check and collection:
            st.markdown(
                """
                <div style="text-align: right;">
                    <span class="status-badge">
                        <span class="status-dot"></span> System Online & Grounded
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

    # 4. Ingestion Section (Collapsible Accordion)
    has_processed = bool(st.session_state.get("last_processing_result"))
    with st.expander("📑 Ingest & Index PDF Document", expanded=(not has_processed)):
        col_upload, col_action = st.columns([0.72, 0.28])
        with col_upload:
            uploaded_file = st.file_uploader(
                label="Choose a PDF file to index into ChromaDB",
                type=["pdf"],
                help="Only PDF (.pdf) format is supported.",
                label_visibility="collapsed"
            )
            if uploaded_file is not None:
                st.caption(f"📁 **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB ready to process)")

        with col_action:
            st.markdown("<div style='height: 0.2rem;'></div>", unsafe_allow_html=True)
            process_clicked = st.button("⚡ Index Document", type="primary", use_container_width=True)

        # Ingestion Logic
        if process_clicked:
            if uploaded_file is None:
                st.warning("⚠️ Please select a PDF file before clicking Index Document.")
            else:
                collection = get_vector_collection()
                if collection is None:
                    st.error("❌ Failed to initialize ChromaDB. Please check directory permissions.")
                else:
                    temp_dir = tempfile.mkdtemp()
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)

                    try:
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        with st.spinner(f"Extracting pages, generating clean chunks, and computing embeddings for '{uploaded_file.name}'..."):
                            # Deduplicate previous records for this file before storing new ones
                            deleted_count = delete_documents_by_source(collection, uploaded_file.name)
                            summary = process_and_store_pdf(temp_file_path, collection)

                        if summary.get("total_pages", 0) == 0:
                            st.error("❌ Could not extract text. The document may be empty, image-only, or password protected.")
                            st.session_state["last_processing_result"] = None
                        elif summary.get("stored_chunks", 0) == 0:
                            st.error("❌ Failed to generate embeddings or store chunks.")
                            st.session_state["last_processing_result"] = None
                        else:
                            st.session_state["last_processing_result"] = {
                                "filename": uploaded_file.name,
                                "total_pages": summary["total_pages"],
                                "total_chunks": summary["total_chunks"],
                                "stored_chunks": summary["stored_chunks"],
                                "deleted_prev": deleted_count
                            }
                            st.rerun()

                    except Exception as e:
                        st.error(f"❌ Ingestion failed: {e}")
                        st.session_state["last_processing_result"] = None
                    finally:
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                        if os.path.exists(temp_dir):
                            os.rmdir(temp_dir)

    # Ingestion Metrics Display
    if st.session_state.get("last_processing_result"):
        res = st.session_state["last_processing_result"]
        del_msg = f" (safely updated {res['deleted_prev']} previous chunks)" if res.get("deleted_prev", 0) > 0 else ""
        st.success(f"✅ **{res['filename']}** is indexed and ready for Q&A{del_msg}!")

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{res['total_pages']}</div>
                    <div class="metric-label">Pages Processed</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m_col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{res['total_chunks']}</div>
                    <div class="metric-label">Chunks Split</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m_col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{res['stored_chunks']}</div>
                    <div class="metric-label">Vectors Stored</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<hr style='margin: 1.2rem 0; opacity: 0.15;'>", unsafe_allow_html=True)

    # 5. Interactive Chat Section
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Display empty state welcome card if no messages yet
    if not st.session_state["chat_history"]:
        st.markdown(
            """
            <div class="welcome-card">
                <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0.4rem; color: #4F46E5;">
                    💡 Getting Started with Grounded Q&A
                </div>
                <div style="font-size: 0.9rem; line-height: 1.5; opacity: 0.85;">
                    1. Ensure your PDF is indexed above using the <b>Index Document</b> button.<br>
                    2. Type any question in the chat input below.<br>
                    3. The assistant will retrieve semantic passages and answer strictly based on the text with verifiable citations.<br>
                    4. Try asking sample questions like: <i>"What is Machine Learning?"</i>, <i>"Explain Supervised Learning"</i>, or <i>"What are the advantages and disadvantages of Machine Learning?"</i>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Render previous conversation history
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

            # Render polished source cards
            if msg.get("sources"):
                with st.expander(f"📚 View {len(msg['sources'])} Grounded Sources & Citations"):
                    for idx, src in enumerate(msg["sources"], 1):
                        source_name = src.get("source", "Unknown Document")
                        page_num = src.get("page_number", "N/A")
                        distance = src.get("distance", 0.0)
                        snippet = src.get("text", "")

                        st.markdown(
                            f"""
                            <div class="citation-card">
                                <div class="citation-header">
                                    <span>
                                        <span class="badge-pill">Source #{idx}</span>
                                        <span style="margin-left: 0.4rem;">{source_name} (Page {page_num})</span>
                                    </span>
                                    <span class="dist-pill">Score: {distance:.4f}</span>
                                </div>
                                <div class="citation-snippet">"{snippet[:320]}..."</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            elif msg["role"] == "assistant" and msg.get("has_query") and not msg.get("sources"):
                st.caption(f"ℹ️ No passages passed the relevance threshold (distance ≤ {dist_threshold}).")

    # 6. Chat Input Handler
    user_query = st.chat_input("Ask a question about your uploaded PDF document...")

    if user_query:
        # Guard 1: Verify collection availability and content
        collection = get_vector_collection()
        if collection is None or collection.count() == 0:
            st.warning("⚠️ Please upload and index a PDF document above before asking questions.")
            return

        # Record and render user query
        st.session_state["chat_history"].append({"role": "user", "content": user_query})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_query)

        # Guard 2: Verify Groq Client availability
        groq_client = get_groq_client()
        if groq_client is None:
            error_msg = (
                "⚠️ **Groq Client is not initialized.**\n\n"
                "Please configure `GROQ_API_KEY` in your `.env` file:\n"
                "```bash\nGROQ_API_KEY=gsk_your_actual_key_here\n```\n"
                "Free API keys are available at [console.groq.com/keys](https://console.groq.com/keys)."
            )
            with st.chat_message("assistant", avatar="🤖"):
                st.error(error_msg)
            st.session_state["chat_history"].append({"role": "assistant", "content": error_msg})
            return

        # Generate Grounded Answer via Groq
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner(f"Retrieving top {top_k} passages and generating grounded answer via Groq ({MODEL_NAME})..."):
                # Semantic vector search with threshold filtering
                search_response = search_pdf(
                    collection=collection,
                    query=user_query,
                    top_k=top_k,
                    distance_threshold=dist_threshold
                )
                retrieved_chunks = search_response.get("results", [])

                # LLM answer generation
                answer = generate_answer(groq_client, user_query, retrieved_chunks)

                if not answer or not answer.strip():
                    answer = "The provided document context does not contain sufficient information to answer this question."

                st.markdown(answer)

                # Render polished source cards
                if retrieved_chunks:
                    with st.expander(f"📚 View {len(retrieved_chunks)} Grounded Sources & Citations"):
                        for idx, src in enumerate(retrieved_chunks, 1):
                            source_name = src.get("source", "Unknown Document")
                            page_num = src.get("page_number", "N/A")
                            distance = src.get("distance", 0.0)
                            snippet = src.get("text", "")

                            st.markdown(
                                f"""
                                <div class="citation-card">
                                    <div class="citation-header">
                                        <span>
                                            <span class="badge-pill">Source #{idx}</span>
                                            <span style="margin-left: 0.4rem;">{source_name} (Page {page_num})</span>
                                        </span>
                                        <span class="dist-pill">Score: {distance:.4f}</span>
                                    </div>
                                    <div class="citation-snippet">"{snippet[:320]}..."</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                else:
                    st.caption(f"ℹ️ No passages passed the relevance threshold (distance ≤ {dist_threshold}).")

            # Save assistant message to session state
            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": answer,
                "sources": retrieved_chunks,
                "has_query": True
            })


if __name__ == "__main__":
    main()
