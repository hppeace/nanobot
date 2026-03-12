"""Text-to-Speech tool for agent."""

import os
import time
from pathlib import Path

import httpx
from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.config.paths import get_media_dir


class TTSTool(Tool):
    """Convert text to speech using DashScope (Alibaba Cloud Qwen TTS)."""

    name = "text_to_speech"
    description = "Convert text into a speech audio file."
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to convert to speech",
            },
        },
        "required": ["text"],
    }

    def __init__(self, api_key: str | None = None):
        self._init_api_key = api_key


    @property
    def api_key(self) -> str:
        """Resolve API key at call time so env/config changes are picked up."""
        return self._init_api_key or os.environ.get("DASHSCOPE_API_KEY", "")

    async def execute(self, text: str) -> str:
        """Execute TTS conversion."""
        if not self.api_key:
            return (
                "Error: DashScope API key not configured. Set it in "
                "~/.nanobot/config.json under tools.tts.apiKey "
                "(or export DASHSCOPE_API_KEY), then restart the gateway."
            )

        if not text or not text.strip():
            return "Error: Empty text provided"

        try:
            # Generate output path
            media_dir = get_media_dir("tts")
            output_path = media_dir / f"speech_{int(time.time() * 1000)}.mp3"

            # Build request
            request_data = {
                "model": "qwen3-tts-flash",
                "input": {
                    "text": text,
                    "voice": "Cherry",
                    "language_type": "Chinese",
                },
            }

            # Call API
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_data,
                )
                response.raise_for_status()
                result = response.json()

                if result.get("code"):
                    error_msg = result.get("message", "Unknown error")
                    logger.error("DashScope TTS error: {}", error_msg)
                    return f"Error: {error_msg}"

                audio_url = result.get("output", {}).get("audio", {}).get("url")
                if not audio_url:
                    return "Error: No audio URL in response"

                # Download audio
                audio_response = await client.get(audio_url)
                audio_response.raise_for_status()
                Path(output_path).write_bytes(audio_response.content)
                logger.info("TTS generated: {}", output_path)
                return f"Speech generated: {output_path}"

        except httpx.HTTPStatusError as e:
            logger.error("TTS HTTP error: {}", e)
            return f"Error: HTTP {e.response.status_code}"
        except Exception as e:
            logger.exception("TTS error")
            return f"Error: {e}"
