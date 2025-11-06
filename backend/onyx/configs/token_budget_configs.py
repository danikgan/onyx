import os

RAG_CHUNKS_TO_LLM = int(os.environ.get("RAG_CHUNKS_TO_LLM") or 5)
RAG_CHUNK_MAX_CHARS = int(os.environ.get("RAG_CHUNK_MAX_CHARS") or 1600)
RAG_TOTAL_MAX_CHARS = int(os.environ.get("RAG_TOTAL_MAX_CHARS") or 8000)
HISTORY_SUMMARY_TARGET_TOKENS = int(
    os.environ.get("HISTORY_SUMMARY_TARGET_TOKENS") or 900
)
TOOL_OUTPUT_MAX_CHARS = int(os.environ.get("TOOL_OUTPUT_MAX_CHARS") or 2000)
MODEL_WINDOW_TOKENS = int(os.environ.get("MODEL_WINDOW_TOKENS") or 128_000)
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS") or 1024)
MIN_OUTPUT_TOKENS = int(os.environ.get("MIN_OUTPUT_TOKENS") or 768)
