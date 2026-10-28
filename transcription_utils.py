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
    if total_ms < 0:
        raise ValueError("total_ms must be non-negative")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be greater than 0")
    if max_chunk_ms <= 0:
        raise ValueError("max_chunk_ms must be greater than 0")
    if min_chunk_ms <= 0:
        raise ValueError("min_chunk_ms must be greater than 0")
    if max_chunk_ms < min_chunk_ms:
        raise ValueError("max_chunk_ms cannot be smaller than min_chunk_ms")
    if bitrate_kbps <= 0:
        raise ValueError("bitrate_kbps must be greater than 0")
    if total_ms == 0:
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
        if midpoint == start_ms or midpoint == end_ms:
            return [(start_ms, end_ms)]

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


SECTION_HEADINGS = {"agenda", "decisions", "action items"}


def minutes_to_html(text):
    """Plain-text minutes -> HTML (bold headings, real bullet lists) for rich-text paste."""
    import html

    out, in_list = [], False
    for line in text.splitlines():
        stripped = line.strip()
        is_bullet = stripped.startswith("- ")
        if in_list and not is_bullet:
            out.append("</ul>")
            in_list = False
        if is_bullet:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{html.escape(stripped[2:])}</li>")
        elif stripped.rstrip(":").lower() in SECTION_HEADINGS:
            out.append(f"<p><b>{html.escape(stripped.rstrip(':'))}</b></p>")
        elif stripped:
            out.append(f"<p>{html.escape(stripped)}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def _normalize(text):
    return " ".join(text.split()).lower()


def generate_new_version(generate, versions, attempts=3):
    """Call generate() until it returns text not already in versions (ignoring whitespace/case).

    Returns None if every attempt duplicated an existing version.
    """
    seen = {_normalize(v) for v in versions}
    for _ in range(attempts):
        candidate = generate()
        if _normalize(candidate) not in seen:
            return candidate
    return None
