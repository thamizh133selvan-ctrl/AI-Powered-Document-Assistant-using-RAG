"""
Answer Generator Module for AI-Powered PDF Assistant.
Groq Answer Generation using Groq Chat Completions API.

This module provides functions to:
1. Accept an initialized Groq client, user query, and retrieved chunks.
2. Build a grounded prompt using prompt_builder.build_prompt().
3. Query the Groq model (openai/gpt-oss-120b) and extract the generated response text.
4. Safely handle errors, invalid inputs, rate limits, and API exceptions without exposing secrets.
"""

from typing import List, Dict, Any, Optional
from prompt_builder import build_prompt

# Configurable default Groq model name
MODEL_NAME: str = "openai/gpt-oss-120b"


def generate_answer(
    client: Any,
    query: str,
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    model: str = MODEL_NAME
) -> str:
    """
    Generates an answer to the user query grounded on retrieved PDF chunks using the Groq API.

    Args:
        client (Any): Initialized Groq Client instance.
        query (str): The user's question or search query.
        retrieved_chunks (Optional[List[Dict[str, Any]]]): List of chunk dictionaries
            retrieved from vector storage. Defaults to None.
        model (str): Groq model identifier. Defaults to MODEL_NAME ("openai/gpt-oss-120b").

    Returns:
        str: Generated answer text. Returns an empty string "" on invalid client,
             empty prompt, API errors, rate limits, or empty responses.
    """
    # 1. Validate the client
    if client is None:
        print("[Error] A valid initialized Groq client must be provided to generate_answer.")
        return ""

    # 2. Build the grounded prompt using prompt_builder
    prompt = build_prompt(query, retrieved_chunks)

    # 3. If prompt is empty (e.g. invalid query), return empty string without calling Groq
    if not prompt or not prompt.strip():
        print("[Notice] Prompt generation produced an empty prompt. Skipping Groq API call.")
        return ""

    # 4. Call Groq Chat Completions API
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # 5. Extract response text safely
        if response is None:
            print("[Warning] Received None response from Groq API.")
            return ""

        choices = getattr(response, "choices", None)
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            print("[Warning] Groq returned response with no choices.")
            return ""

        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            print("[Warning] Groq choice contains no message.")
            return ""

        answer_text = getattr(message, "content", None)
        if not answer_text or not isinstance(answer_text, str) or not answer_text.strip():
            print("[Warning] Groq returned an empty response content.")
            return ""

        return answer_text.strip()

    except Exception as e:
        # Sanitize output to prevent any accidental exposure of sensitive parameters/keys
        error_type = type(e).__name__
        print(f"[Error] Groq API generation failed ({error_type}): {e}")
        return ""
