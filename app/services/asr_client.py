"""ASR API client for ChatGLM autoglm-asr-vllm interface."""

import logging
from typing import Optional

import httpx

from app.config import CHATGLM_API_KEY, CHATGLM_BASE_URL, CHATGLM_MODEL, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)


class ASRClient:
    """Client for interacting with ChatGLM ASR API."""

    def __init__(self):
        self.api_key = CHATGLM_API_KEY
        self.base_url = CHATGLM_BASE_URL
        self.model = CHATGLM_MODEL
        self.timeout = REQUEST_TIMEOUT
        logger.info(f"ASRClient initialized with base_url={self.base_url}, model={self.model}")

    def _build_request_body(self, audio_data_url: str, prompt: Optional[str] = None) -> dict:
        """Build the request body for ASR API.

        Args:
            audio_data_url: Base64 encoded audio data URL
            prompt: Optional prompt for the transcription

        Returns:
            Request body dictionary
        """
        content = [
            {
                "type": "text",
                "text": "<|begin_of_audio|><|endoftext|><|end_of_audio|>"
            },
            {
                "type": "audio_url",
                "audio_url": {
                    "url": audio_data_url
                }
            }
        ]

        if prompt:
            content.append({
                "type": "text",
                "text": prompt
            })
        else:
            content.append({
                "type": "text",
                "text": "将这段音频转录成文字。"
            })

        return {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个名为 ChatGLM 的人工智能助手。"
                },
                {
                    "role": "user",
                    "content": content
                }
            ]
        }

    async def transcribe(self, audio_data_url: str, prompt: Optional[str] = None) -> str:
        """Transcribe audio data to text.

        Args:
            audio_data_url: Base64 encoded audio data URL
            prompt: Optional prompt for the transcription

        Returns:
            Transcribed text

        Raises:
            ValueError: If API key is not configured
            httpx.HTTPStatusError: For authentication or HTTP errors
            httpx.TimeoutException: For timeout errors
            Exception: For other errors
        """
        if not self.api_key:
            logger.error("CHATGLM_API_KEY is not configured")
            raise ValueError("CHATGLM_API_KEY is not configured")

        request_body = self._build_request_body(audio_data_url, prompt)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        url = f"{self.base_url}/chat/completions"
        logger.info(f"ASR API request to {url}")
        logger.debug(f"Request body: {request_body}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                url,
                headers=headers,
                json=request_body
            )

            logger.info(f"ASR API response status: {response.status_code}")
            logger.info(f"ASR API response body: {response.text}")

            if response.status_code == 401 or response.status_code == 403:
                logger.error(f"ASR API authentication failed: {response.status_code}")
                raise httpx.HTTPStatusError(
                    "Authentication failed. Please check your API key.",
                    request=response.request,
                    response=response
                )

            response.raise_for_status()

            data = response.json()
            logger.debug(f"ASR API response data: {data}")

            # Extract transcript from response
            if "choices" in data and len(data["choices"]) > 0:
                message = data["choices"][0].get("message", {})
                content = message.get("content", "")
                logger.info(f"Transcribed text: {content[:100]}..." if len(content) > 100 else f"Transcribed text: {content}")
                return content.strip()
            else:
                logger.error(f"Unexpected response format: {data}")
                raise ValueError(f"Unexpected response format: {data}")


# Singleton instance
asr_client = ASRClient()
