import re


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


# Free Groq chat models, best first. Availability varies per API key/tier.
PREFERRED_CHAT_MODELS = (
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
)

# Model ids that are not chat completion models.
_NON_CHAT_MARKERS = ("whisper", "tts", "guard")


def choose_chat_model(available, override=None):
    """Pick a chat model this API key can actually call.

    `available` is the set of model ids returned by the provider.
    """
    if override:
        return override

    available = set(available)
    for model in PREFERRED_CHAT_MODELS:
        if model in available:
            return model

    chat_only = sorted(
        m for m in available
        if not any(marker in m.lower() for marker in _NON_CHAT_MARKERS)
    )
    if not chat_only:
        raise RuntimeError("No chat-capable model available for this API key.")
    return chat_only[0]


def strip_markdown(text):
    """Strip markdown markers so the minutes read as plain corporate text.

    The model still sneaks in `**bold**` and `## headings` however firmly the
    prompt says not to, so remove them here rather than trusting the prompt.
    """
    if not text:
        return text

    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)      # ## Heading
    text = re.sub(r"^(\s*)[*+]\s+", r"\1- ", text, flags=re.MULTILINE)  # * bullet -> - bullet
    text = re.sub(r"\*{1,3}(\S.*?\S|\S)\*{1,3}", r"\1", text)          # **bold** / *italic*
    text = text.replace("*", "").replace("#", "")                       # leftovers
    return text.strip()
