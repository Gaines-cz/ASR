"""Transcript text merging utilities."""

from app.config import MERGE_MIN_OVERLAP_CHARS


def merge_chunks(chunks: dict) -> str:
    """Merge transcribed chunks into a single text with deduplication.

    Args:
        chunks: Dictionary mapping chunk_index to transcribed text

    Returns:
        Merged text from all chunks
    """
    if not chunks:
        return ""

    # Sort chunks by index
    sorted_items = sorted(chunks.items())

    # Filter out empty chunks
    non_empty = [text.strip() for _, text in sorted_items if text.strip()]

    if not non_empty:
        return ""

    # Simple case: single chunk
    if len(non_empty) == 1:
        return non_empty[0]

    # Detect overlaps between adjacent chunks and deduplicate
    result = [non_empty[0]]

    for i in range(1, len(non_empty)):
        prev_text = result[-1]
        curr_text = non_empty[i]

        # Find overlap between prev_text ending and curr_text beginning
        overlap = _find_overlap(prev_text, curr_text)

        if overlap:
            # Remove overlapping part from new text to avoid duplication
            result.append(curr_text[len(overlap):])
        else:
            result.append(curr_text)

    return " ".join(result)


def _find_overlap(s1: str, s2: str, min_match: int = None) -> str:
    """Find the overlapping suffix/prefix between two strings.

    Args:
        s1: First string (we check its suffix against s2's prefix)
        s2: Second string
        min_match: Minimum overlap length (default from config)

    Returns:
        The overlapping string, or empty string if no significant overlap
    """
    if min_match is None:
        min_match = MERGE_MIN_OVERLAP_CHARS

    if not s1 or not s2:
        return ""

    result = ""
    # Check s1's suffix against s2's prefix
    for i in range(1, min(len(s1), len(s2)) + 1):
        if s1[-i:] == s2[:i]:
            result = s2[:i]

    return result if len(result) >= min_match else ""
