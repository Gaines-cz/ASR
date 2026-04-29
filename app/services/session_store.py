"""Session management for real-time transcription."""

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

from app.services.transcript_merge import merge_chunks


SESSION_TIMEOUT = 300  # 5 minutes


@dataclass
class Session:
    """Represents a transcription session with multiple chunks."""

    session_id: str
    chunks: Dict[int, str] = field(default_factory=dict)
    audio_segments: list = field(default_factory=list)  # Audio segment metadata
    window_config: dict = field(default_factory=dict)    # Window configuration
    last_update: float = field(default_factory=time.time)

    def add_chunk(self, chunk_index: int, text: str) -> None:
        """Add a transcribed chunk to the session.

        Args:
            chunk_index: Index of the chunk
            text: Transcribed text for this chunk
        """
        self.chunks[chunk_index] = text
        self.last_update = time.time()

    def add_audio_segment(self, chunk_index: int, duration_ms: int) -> None:
        """Add audio segment metadata.

        Args:
            chunk_index: Index of the chunk
            duration_ms: Duration of the audio segment in milliseconds
        """
        self.audio_segments.append({
            'chunk_index': chunk_index,
            'duration_ms': duration_ms
        })

    def set_window_config(self, window_size_ms: int, overlap_ms: int) -> None:
        """Set the window configuration.

        Args:
            window_size_ms: Window size in milliseconds
            overlap_ms: Overlap size in milliseconds
        """
        self.window_config = {
            'window_size_ms': window_size_ms,
            'overlap_ms': overlap_ms
        }

    def get_merged_text(self) -> str:
        """Get all chunks merged into a single text.

        Returns:
            Merged text from all chunks
        """
        return merge_chunks(self.chunks)

    def finalize(self) -> str:
        """Finalize the session and return the complete transcript.

        Returns:
            Final merged transcript
        """
        return self.get_merged_text()


# In-memory session storage
sessions: Dict[str, Session] = {}
_sessions_lock = threading.Lock()


def _cleanup_expired_sessions() -> None:
    """Background thread to clean up expired sessions."""
    while True:
        time.sleep(60)
        now = time.time()
        with _sessions_lock:
            expired = [
                sid for sid, session in sessions.items()
                if now - session.last_update > SESSION_TIMEOUT
            ]
            for sid in expired:
                del sessions[sid]
                print(f"[SESSION] Cleaned up expired session: {sid}")


# Start cleanup thread
_cleanup_thread = threading.Thread(target=_cleanup_expired_sessions, daemon=True)
_cleanup_thread.start()


def create_session() -> str:
    """Create a new transcription session.

    Returns:
        The session ID for the new session
    """
    session_id = str(uuid.uuid4())
    with _sessions_lock:
        sessions[session_id] = Session(session_id=session_id)
    return session_id


def add_chunk(session_id: str, chunk_index: int, text: str, window_size_ms: int = None, overlap_ms: int = None, duration_ms: int = None) -> Session:
    """Add a chunk to an existing session.

    Args:
        session_id: The session ID
        chunk_index: Index of the chunk
        text: Transcribed text
        window_size_ms: Window size in milliseconds (optional)
        overlap_ms: Overlap size in milliseconds (optional)
        duration_ms: Duration of audio segment in milliseconds (optional)

    Returns:
        The updated session

    Raises:
        KeyError: If session not found
    """
    with _sessions_lock:
        if session_id not in sessions:
            # Session was finalized but async chunk still arrived - return success
            print(f"[SESSION] Chunk {chunk_index} arrived for finalized session {session_id}, ignoring")
            return None

        session = sessions[session_id]
        session.add_chunk(chunk_index, text)

        # Store window config if provided
        if window_size_ms is not None and overlap_ms is not None:
            session.set_window_config(window_size_ms, overlap_ms)

        # Store audio segment metadata if provided
        if duration_ms is not None:
            session.add_audio_segment(chunk_index, duration_ms)

        return session


def get_merged_text(session_id: str) -> str:
    """Get the merged text for a session.

    Args:
        session_id: The session ID

    Returns:
        Merged text from all chunks

    Raises:
        KeyError: If session not found
    """
    with _sessions_lock:
        if session_id not in sessions:
            raise KeyError(f"Session not found: {session_id}")

        return sessions[session_id].get_merged_text()


def finalize_session(session_id: str) -> str:
    """Finalize a session and return the complete transcript.

    Args:
        session_id: The session ID

    Returns:
        Final merged transcript

    Raises:
        KeyError: If session not found
    """
    with _sessions_lock:
        if session_id not in sessions:
            raise KeyError(f"Session not found: {session_id}")

        final_text = sessions[session_id].finalize()

        # Clean up session after finalization
        del sessions[session_id]

        return final_text


def get_session(session_id: str) -> Optional[Session]:
    """Get a session by ID.

    Args:
        session_id: The session ID

    Returns:
        The session if found, None otherwise
    """
    with _sessions_lock:
        return sessions.get(session_id)
