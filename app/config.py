"""Configuration management for ASR application."""

import os
from pathlib import Path


def _load_api_key() -> str:
    """Load API key from environment variable or glm_api_key file."""
    # Try environment variable first
    api_key = os.getenv("CHATGLM_API_KEY")
    if api_key:
        return api_key

    # Try glm_api_key file in project root
    key_file = Path(__file__).parent.parent / "glm_api_key"
    if key_file.exists():
        content = key_file.read_text().strip()
        # If it's a KEY = "value" format, extract just the value
        if "=" in content:
            api_key = content.split("=", 1)[1].strip().strip('"').strip("'")
            return api_key
        return content

    return ""


# ChatGLM API Configuration
CHATGLM_API_KEY = _load_api_key()
CHATGLM_BASE_URL = "https://api.chatglm.cn/v1"
CHATGLM_MODEL = "autoglm-asr-vllm"
REQUEST_TIMEOUT = 120

# Sliding Window Configuration
SLIDING_WINDOW_SIZE_MS = 3000    # 3 second window
SLIDING_OVERLAP_MS = 1500        # 50% overlap

# VAD Configuration
VAD_ENERGY_THRESHOLD = 0.01
VAD_SPEECH_END_FRAMES = 15       # ~750ms silence to confirm speech end
VAD_MIN_SPEECH_MS = 300          # minimum speech duration

# Merge Configuration
MERGE_MIN_OVERLAP_CHARS = 3      # minimum overlap characters for deduplication
