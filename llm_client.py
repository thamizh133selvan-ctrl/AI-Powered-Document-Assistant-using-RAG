"""
LLM Client Module for AI-Powered PDF Assistant.
Groq API Client Initialization using the groq Python SDK.

This module provides modular, beginner-friendly functions to:
1. Load environment variables securely from .env using python-dotenv.
2. Read and validate the GROQ_API_KEY environment variable.
3. Initialize and return the Groq Client safely with error handling.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from groq import Groq


def initialize_groq(api_key: Optional[str] = None) -> Optional[Groq]:
    """
    Initializes and returns a Groq Client instance.

    Loads the GROQ_API_KEY from environment variables (including .env file)
    unless an explicit api_key parameter is provided.

    Args:
        api_key (Optional[str]): Optional Groq API key string.
            If None, reads from os.getenv("GROQ_API_KEY").

    Returns:
        Optional[Groq]: The initialized Groq Client instance,
            or None if the API key is missing, empty, or initialization fails.
    """
    # 1. Load environment variables from the project's .env file
    load_dotenv()

    # 2. Retrieve the API key from argument or environment variable
    key_to_use = api_key if api_key is not None else os.getenv("GROQ_API_KEY")

    # 3. Check whether the API key exists and is non-empty
    if not key_to_use or not isinstance(key_to_use, str) or not key_to_use.strip():
        print(
            "[Error] GROQ_API_KEY environment variable not found or empty.\n"
            "        Please add your Groq API key to the .env file:\n"
            "        GROQ_API_KEY=your_actual_api_key_here"
        )
        return None

    # Check for unreplaced placeholder
    if key_to_use.strip() == "your_groq_api_key_here":
        print(
            "[Error] GROQ_API_KEY in .env is still set to placeholder 'your_groq_api_key_here'.\n"
            "        Please replace it with your actual Groq API key from https://console.groq.com/keys"
        )
        return None

    # 4. Safely initialize the Groq Client
    try:
        client = Groq(api_key=key_to_use.strip())
        return client
    except Exception as e:
        print(f"[Error] Failed to initialize Groq Client: {e}")
        return None


# Backward-compatible generic alias
initialize_llm = initialize_groq
