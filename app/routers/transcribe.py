"""Transcription API routes."""

import logging
from typing import Optional

import httpx
from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

from app.services.asr_client import asr_client
from app.services import session_store
from app.utils.audio import detect_audio_format, encode_audio_to_data_url, validate_audio_format, convert_to_wav

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/transcribe", tags=["transcribe"])


class FileTranscribeResponse(BaseModel):
    """Response for file transcription."""
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    error_code: Optional[str] = None


class ChunkTranscribeResponse(BaseModel):
    """Response for chunk transcription."""
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    error_code: Optional[str] = None


class FinalizeResponse(BaseModel):
    """Response for finalize."""
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    error_code: Optional[str] = None


@router.post("/file", response_model=FileTranscribeResponse)
async def transcribe_file(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None)
):
    """Transcribe an audio file.

    Args:
        file: The audio file to transcribe
        prompt: Optional prompt for transcription

    Returns:
        Transcription result
    """
    logger.info(f"Received file transcription request: filename={file.filename}, content_type={file.content_type}")

    # Check if file is provided
    if not file:
        logger.warning("No file provided in request")
        return FileTranscribeResponse(
            success=False,
            error="No file provided",
            error_code="EMPTY_AUDIO_FILE"
        )

    # Read file content
    try:
        audio_bytes = await file.read()
        logger.info(f"Read {len(audio_bytes)} bytes from file")
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        return FileTranscribeResponse(
            success=False,
            error=f"Failed to read file: {str(e)}",
            error_code="FILE_READ_ERROR"
        )

    # Check if file is empty
    if not audio_bytes:
        logger.warning("Empty file provided")
        return FileTranscribeResponse(
            success=False,
            error="Empty file",
            error_code="EMPTY_AUDIO_FILE"
        )

    # Detect audio format
    mime_type = file.content_type or "audio/wav"
    format_name = detect_audio_format(file.filename or "audio.wav", mime_type)
    logger.info(f"Detected audio format: {format_name}")

    # Validate format
    if not validate_audio_format(format_name):
        logger.warning(f"Unsupported audio format: {format_name}")
        return FileTranscribeResponse(
            success=False,
            error=f"Unsupported audio format: {format_name}",
            error_code="UNSUPPORTED_AUDIO_FORMAT"
        )

    # Encode to data URL (convert to wav if needed)
    try:
        # ASR model only supports wav, convert if necessary
        if format_name != "wav":
            logger.info(f"Converting {format_name} to wav...")
            audio_bytes = convert_to_wav(audio_bytes, format_name)
            format_name = "wav"
            logger.info(f"Conversion complete, new size: {len(audio_bytes)} bytes")

        audio_data_url = encode_audio_to_data_url(audio_bytes, format_name)
        logger.debug(f"Encoded audio to data URL, length: {len(audio_data_url)}")
    except ImportError as e:
        logger.error(f"Audio conversion failed: {e}")
        return FileTranscribeResponse(
            success=False,
            error=str(e),
            error_code="AUDIO_CONVERT_ERROR"
        )
    except Exception as e:
        logger.error(f"Failed to encode audio: {e}")
        return FileTranscribeResponse(
            success=False,
            error=f"Failed to encode audio: {str(e)}",
            error_code="AUDIO_ENCODE_ERROR"
        )

    # Call ASR API
    try:
        logger.info("Calling ASR API for file transcription...")
        transcript = await asr_client.transcribe(audio_data_url, prompt)
        logger.info(f"File transcription successful, result length: {len(transcript)}")
        return FileTranscribeResponse(
            success=True,
            data={
                "transcript": transcript,
                "format": format_name
            }
        )
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return FileTranscribeResponse(
            success=False,
            error=str(e),
            error_code="CONFIG_ERROR"
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"ASR API HTTP error: {e.response.status_code}")
        if e.response.status_code in (401, 403):
            return FileTranscribeResponse(
                success=False,
                error="Authentication failed. Please check your API key.",
                error_code="ASR_AUTH_FAILED"
            )
        return FileTranscribeResponse(
            success=False,
            error=f"ASR API error: {e.response.status_code}",
            error_code="ASR_REQUEST_FAILED"
        )
    except httpx.TimeoutException:
        logger.error("ASR API request timed out")
        return FileTranscribeResponse(
            success=False,
            error="Request to ASR API timed out",
            error_code="ASR_TIMEOUT"
        )
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return FileTranscribeResponse(
            success=False,
            error=f"Transcription failed: {str(e)}",
            error_code="ASR_REQUEST_FAILED"
        )


@router.post("/session")
async def create_session():
    """Create a new transcription session for real-time recording.

    Returns:
        Session ID for the new session
    """
    session_id = session_store.create_session()
    logger.info(f"Created new transcription session: {session_id}")
    return {
        "success": True,
        "data": {
            "session_id": session_id
        }
    }


@router.post("/chunk", response_model=ChunkTranscribeResponse)
async def transcribe_chunk(
    session_id: str = Form(...),
    chunk_index: int = Form(...),
    file: UploadFile = File(...),
    mime_type: str = Form("audio/webm"),
    window_size_ms: int = Form(3000),
    overlap_ms: int = Form(1500)
):
    """Transcribe an audio chunk from real-time recording.

    Args:
        session_id: The session ID
        chunk_index: Index of the chunk
        file: The audio chunk file
        mime_type: MIME type of the audio
        window_size_ms: Sliding window size in milliseconds
        overlap_ms: Sliding window overlap in milliseconds

    Returns:
        Transcription result with partial and merged text
    """
    logger.info(f"Received chunk transcription: session_id={session_id}, chunk_index={chunk_index}, filename={file.filename}, mime_type={mime_type}, window={window_size_ms}ms, overlap={overlap_ms}ms")

    # Read file content
    try:
        audio_bytes = await file.read()
        logger.info(f"Read {len(audio_bytes)} bytes from chunk")
    except Exception as e:
        logger.error(f"Failed to read chunk: {e}")
        return ChunkTranscribeResponse(
            success=False,
            error=f"Failed to read chunk: {str(e)}",
            error_code="FILE_READ_ERROR"
        )

    # Check if file is empty
    if not audio_bytes:
        logger.warning("Empty chunk provided")
        return ChunkTranscribeResponse(
            success=False,
            error="Empty chunk",
            error_code="EMPTY_AUDIO_FILE"
        )

    # Detect audio format
    format_name = detect_audio_format(file.filename or "chunk.webm", mime_type)
    logger.info(f"Detected audio format: {format_name}")

    # Validate format
    if not validate_audio_format(format_name):
        logger.warning(f"Unsupported audio format: {format_name}")
        return ChunkTranscribeResponse(
            success=False,
            error=f"Unsupported audio format: {format_name}",
            error_code="UNSUPPORTED_AUDIO_FORMAT"
        )

    # Encode to data URL (convert to wav if needed)
    try:
        # ASR model only supports wav, convert if necessary
        if format_name != "wav":
            logger.info(f"Converting {format_name} to wav...")
            audio_bytes = convert_to_wav(audio_bytes, format_name)
            format_name = "wav"
            logger.info(f"Conversion complete, new size: {len(audio_bytes)} bytes")

        audio_data_url = encode_audio_to_data_url(audio_bytes, format_name)
        logger.debug(f"Encoded audio to data URL, length: {len(audio_data_url)}")
    except ImportError as e:
        logger.error(f"Audio conversion failed: {e}")
        return ChunkTranscribeResponse(
            success=False,
            error=str(e),
            error_code="AUDIO_CONVERT_ERROR"
        )
    except Exception as e:
        logger.error(f"Failed to encode audio: {e}")
        return ChunkTranscribeResponse(
            success=False,
            error=f"Failed to encode audio: {str(e)}",
            error_code="AUDIO_ENCODE_ERROR"
        )

    # Call ASR API
    try:
        logger.info(f"Calling ASR API for chunk {chunk_index}...")
        transcript = await asr_client.transcribe(audio_data_url)
        logger.info(f"Chunk {chunk_index} transcription successful: {transcript[:50]}..." if len(transcript) > 50 else f"Chunk {chunk_index} transcription successful: {transcript}")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return ChunkTranscribeResponse(
            success=False,
            error=str(e),
            error_code="CONFIG_ERROR"
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"ASR API HTTP error: {e.response.status_code}")
        if e.response.status_code in (401, 403):
            return ChunkTranscribeResponse(
                success=False,
                error="Authentication failed. Please check your API key.",
                error_code="ASR_AUTH_FAILED"
            )
        return ChunkTranscribeResponse(
            success=False,
            error=f"ASR API error: {e.response.status_code}",
            error_code="ASR_REQUEST_FAILED"
        )
    except httpx.TimeoutException:
        logger.error("ASR API request timed out")
        return ChunkTranscribeResponse(
            success=False,
            error="Request to ASR API timed out",
            error_code="ASR_TIMEOUT"
        )
    except Exception as e:
        return ChunkTranscribeResponse(
            success=False,
            error=f"Transcription failed: {str(e)}",
            error_code="ASR_REQUEST_FAILED"
        )

    # Add chunk to session
    try:
        # Calculate approximate audio duration based on window size
        duration_ms = window_size_ms
        result = session_store.add_chunk(session_id, chunk_index, transcript, window_size_ms, overlap_ms, duration_ms)
        if result is None:
            # Session was already finalized, ignore this chunk
            logger.info(f"Session {session_id} already finalized, ignoring chunk {chunk_index}")
            return ChunkTranscribeResponse(
                success=True,
                data={
                    "chunk_index": chunk_index,
                    "partial_text": transcript,
                    "merged_text": ""
                }
            )
        logger.info(f"Added chunk {chunk_index} to session {session_id}")
    except KeyError:
        logger.error(f"Session not found: {session_id}")
        return ChunkTranscribeResponse(
            success=False,
            error=f"Session not found: {session_id}",
            error_code="SESSION_NOT_FOUND"
        )

    # Get merged text
    merged_text = session_store.get_merged_text(session_id)
    logger.info(f"Session {session_id} merged text length: {len(merged_text)}")

    return ChunkTranscribeResponse(
        success=True,
        data={
            "chunk_index": chunk_index,
            "partial_text": transcript,
            "merged_text": merged_text
        }
    )


@router.post("/finalize", response_model=FinalizeResponse)
async def finalize_session(session_id: str = Form(...)):
    """Finalize a transcription session and get the complete transcript.

    Args:
        session_id: The session ID to finalize

    Returns:
        Final transcription result
    """
    logger.info(f"Finalizing session: {session_id}")
    try:
        final_text = session_store.finalize_session(session_id)
        logger.info(f"Session {session_id} finalized, final text length: {len(final_text)}")
        return FinalizeResponse(
            success=True,
            data={
                "final_text": final_text
            }
        )
    except KeyError:
        logger.error(f"Session not found during finalize: {session_id}")
        return FinalizeResponse(
            success=False,
            error=f"Session not found: {session_id}",
            error_code="SESSION_NOT_FOUND"
        )
