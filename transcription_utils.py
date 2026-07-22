def estimate_bytes_for_ms(duration_ms: int, bitrate_kbps: int) -> int:
    """Return a rough encoded size estimate for an audio clip in bytes."""
    bytes_per_second = (bitrate_kbps * 1000) // 8
    return max(1, duration_ms * bytes_per_second // 1000)


def get_chunk_windows(
    total_ms: int,
    max_bytes: int,
    max_chunk_ms: int,
    bitrate_kbps: int,
    min_chunk_ms: int,
):
    """Split a total duration into time windows that should stay within a byte budget."""
    if total_ms <= 0:
        return []

    def split(start_ms: int, end_ms: int):
        duration_ms = end_ms - start_ms
        if duration_ms <= 0:
            return []

        if duration_ms <= max_chunk_ms and estimate_bytes_for_ms(duration_ms, bitrate_kbps) <= max_bytes:
            return [(start_ms, end_ms)]

        if duration_ms <= min_chunk_ms:
            return [(start_ms, end_ms)]

        midpoint = start_ms + (duration_ms // 2)
        return split(start_ms, midpoint) + split(midpoint, end_ms)

    return split(0, total_ms)
