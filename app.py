import streamlit as st
from groq import Groq
import subprocess
import tempfile
import os
import imageio_ffmpeg
import json
from dotenv import load_dotenv
try:
    from pydub import AudioSegment
except Exception as e:
    st.error(
        "Audio processing dependency failure: pydub failed to import. "
        "This environment may be missing the native `audioop` module required by pydub. "
        "Ensure you are running CPython with the audioop extension available, or install/run this app in an environment that includes ffmpeg and pydub.\n"
        f"Import error: {e}"
    )
    st.stop()

from transcription_utils import (
    choose_chat_model,
    generate_new_version,
    get_chunk_windows,
    minutes_to_html,
    strip_markdown,
)

# Use the ffmpeg binary shipped by imageio-ffmpeg so no system package is needed.
# Streamlit Cloud's apt step is broken by an expired Debian repo, so packages.txt
# cannot be used to install ffmpeg there.
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
AudioSegment.converter = FFMPEG

# Groq free-tier upload limit is 25 MB. Stay safely under it.
MAX_UPLOAD_BYTES = 24 * 1024 * 1024
# Keep chunks conservative to avoid the 413 request-too-large error.
MAX_CHUNK_MS = 10 * 60 * 1000
MIN_CHUNK_MS = 60 * 1000
TARGET_BITRATE_KBPS = 32

# ---------------- Load .env ----------------
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    st.error("GROQ_API_KEY not found in .env file. Please check your .env file.")
    st.stop()

# Initialize Groq client
client = Groq(api_key=API_KEY)

@st.cache_data(ttl=3600, show_spinner=False)
def pick_chat_model():
    """First available Groq chat model for this key (cached for an hour)."""
    return choose_chat_model(
        (m.id for m in client.models.list().data),
        override=os.getenv("GROQ_CHAT_MODEL"),
    )


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="AI Meeting Minutes Generator", layout="centered")
st.title("🎙️ AI Meeting Minutes Generator")
st.write("Upload any meeting recording and get professional Meeting Minutes instantly.")

# ---------------- Transcription ----------------
def _transcribe_file(file_path):
    """Send one audio file to Groq Whisper and return its text."""
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file
        )
    return transcript.text


def _decode_to_mono_16k(file_path, tmpdir):
    """Decode any supported upload to a 16 kHz mono wav and load it.

    Going through wav on disk keeps pydub on its stdlib `wave` path, which needs no
    ffprobe - imageio-ffmpeg ships ffmpeg only.
    """
    wav_path = os.path.join(tmpdir, "decoded.wav")
    result = subprocess.run(
        [FFMPEG, "-y", "-i", file_path, "-vn", "-ac", "1", "-ar", "16000", wav_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg could not decode this recording, so it cannot be compressed for "
            f"upload. ffmpeg said: {result.stderr.strip()[-500:]}"
        )
    return AudioSegment.from_file(wav_path, format="wav")


def transcribe_audio(file_path):
    """Transcribe. Only decode/compress when the file is over the upload limit.

    Groq accepts mp3/m4a/wav/ogg/mp4 directly, so a small file needs no ffmpeg.
    """
    original_size_bytes = os.path.getsize(file_path)

    if original_size_bytes <= MAX_UPLOAD_BYTES:
        return _transcribe_file(file_path)

    tmpdir = os.path.dirname(file_path)
    audio = _decode_to_mono_16k(file_path, tmpdir)
    compressed_path = os.path.join(tmpdir, "compressed.mp3")
    audio.export(compressed_path, format="mp3", bitrate=f"{TARGET_BITRATE_KBPS}k").close()
    compressed_size_bytes = os.path.getsize(compressed_path)

    if compressed_size_bytes <= MAX_UPLOAD_BYTES:
        return _transcribe_file(compressed_path)

    windows = get_chunk_windows(
        total_ms=len(audio),
        max_bytes=MAX_UPLOAD_BYTES,
        max_chunk_ms=MAX_CHUNK_MS,
        bitrate_kbps=TARGET_BITRATE_KBPS,
        min_chunk_ms=MIN_CHUNK_MS,
    )
    if not windows:
        raise ValueError("Unable to prepare audio for transcription")

    parts = []
    for i, (start_ms, end_ms) in enumerate(windows):
        chunk = audio[start_ms:end_ms]
        chunk_path = os.path.join(tmpdir, f"chunk_{i}.mp3")
        chunk.export(chunk_path, format="mp3", bitrate=f"{TARGET_BITRATE_KBPS}k").close()
        parts.append(_transcribe_file(chunk_path))

    return " ".join(parts)

# ---------------- Corporate MoM Generation ----------------
def generate_minutes(transcript, temperature=0.3):
    prompt = f"""
You are a professional assistant. Summarize the meeting transcript below into exactly three sections, in this order, and nothing else:

Agenda
Decisions
Action Items

Transcript:
{transcript}

Instructions:
- Output ONLY these three sections. No title, heading, preamble, closing remarks, or any other section.
- Agenda, Decisions, and Action Items must all be in bullet points.
- Each Agenda bullet should summarize one discussion point in a full sentence.
- Do NOT include 'Next Steps', 'Adjournment', 'Next Meeting', 'Minutes Prepared by', or any placeholders.
- Maintain professional corporate formatting.
- Output PLAIN TEXT only. No markdown whatsoever: no asterisks (*), no hash (#) headings, no bold, no italics.
- Use a hyphen (-) for every bullet point.
"""

    response = client.chat.completions.create(
        model=pick_chat_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return strip_markdown(response.choices[0].message.content)

# ---------------- User Inputs ----------------
uploaded_file = st.file_uploader(
    "Upload meeting recording",
    type=["mp3", "wav", "m4a", "ogg", "mp4"]
)

# ---------------- Process Upload ----------------
if uploaded_file:
    st.audio(uploaded_file)

    # Every widget click reruns this script, so transcript and versions live in
    # session_state; only a new upload triggers transcription again.
    if st.session_state.get("file_id") != uploaded_file.file_id:
        # ignore_cleanup_errors: on Windows ffmpeg/antivirus can still hold a handle
        # when the block exits, and a failed rmtree would otherwise mask the real error.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            input_path = os.path.join(tmpdir, uploaded_file.name)

            # Save uploaded file temporarily
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            with st.spinner("Transcribing audio using Groq Whisper..."):
                try:
                    transcript = transcribe_audio(input_path)
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    st.stop()

        with st.spinner("Generating professional Meeting Minutes..."):
            try:
                first_version = generate_minutes(transcript)
            except Exception as e:
                st.error(f"Meeting minutes generation failed: {e}")
                st.stop()

        st.session_state.update(
            file_id=uploaded_file.file_id,
            transcript=transcript,
            versions=[first_version],
            version=0,
        )

    transcript = st.session_state.transcript
    versions = st.session_state.versions

    st.subheader("📄 Transcript")
    st.text_area(
        label="Transcript Output",
        value=transcript,
        height=220,
        label_visibility="collapsed",
        disabled=True,
    )

    st.markdown("---")
    st.subheader("📝 Meeting Minutes")

    version_col, regen_col = st.columns([3, 1], vertical_alignment="bottom")

    # ---------------- Regenerate ----------------
    if regen_col.button("🔄 Regenerate", width="stretch"):
        with st.spinner("Generating a new version..."):
            try:
                # Higher temperature so the model actually rephrases; exact repeats are retried.
                new_version = generate_new_version(
                    lambda: generate_minutes(transcript, temperature=0.8), versions
                )
            except Exception as e:
                st.error(f"Meeting minutes generation failed: {e}")
                new_version = None
            else:
                if new_version is None:
                    st.warning("The model kept returning an existing version. Try again.")
        if new_version is not None:
            versions.append(new_version)
            st.session_state.version = len(versions) - 1

    version_col.radio(
        "Version",
        options=range(len(versions)),
        format_func=lambda i: f"v{i + 1}",
        key="version",
        horizontal=True,
    )
    version_index = st.session_state.version

    # Editable: Copy and Download use what the user sees, and edits survive version switches.
    minutes = st.text_area(
        label="Meeting Minutes Output",
        value=versions[version_index],
        height=320,
        label_visibility="collapsed",
        key=f"minutes_v{version_index}",
    )
    versions[version_index] = minutes

    # ---------------- Copy Button ----------------
    # Writes HTML (bold headings, bullet lists) plus plain text, so pasting into
    # Outlook/Word/Teams keeps formatting and plain editors still get clean text.
    copy_col, download_col = st.columns(2)
    payload = json.dumps({"html": minutes_to_html(minutes), "text": minutes}).replace("<", "\\u003c")
    copy_col.iframe(
        f"""
<style>
body {{ margin: 0; overflow: hidden; }}
#copy {{ width: 100%; height: 40px; padding: 0 14px; border: 1px solid rgba(128,128,128,.4); border-radius: 8px; background: transparent;
  color: var(--fg, #31333f); cursor: pointer; font: 13px "Source Sans", "Source Sans Pro", sans-serif; /* iframe falls back to wider font; 13px matches 14px Source Sans */ }}
#copy:hover {{ border-color: #ff4b4b; color: #ff4b4b; }}
</style>
<button id="copy">📋 Copy Minutes</button>
<script>
const data = {payload};
const btn = document.getElementById("copy");
// The iframe is same-origin, so copy the app's live text color: the button follows the
// Streamlit theme, not the OS one. Polled because a theme switch in Settings doesn't rerun.
const syncColor = () => {{
  try {{ btn.style.setProperty("--fg", getComputedStyle(parent.document.querySelector(".stApp")).color); }} catch (e) {{}}
}};
syncColor();
setInterval(syncColor, 500);
btn.onclick = async () => {{
  try {{
    await navigator.clipboard.write([new ClipboardItem({{
      "text/html": new Blob([data.html], {{type: "text/html"}}),
      "text/plain": new Blob([data.text], {{type: "text/plain"}})
    }})]);
    btn.textContent = "✅ Copied";
  }} catch (e) {{
    btn.textContent = "❌ Copy failed";
  }}
  setTimeout(() => btn.textContent = "📋 Copy Minutes", 2000);
}};
</script>
""",
        height=40,
    )

    # ---------------- Download Button ----------------
    download_col.download_button(
        label="📥 Download MoM as TXT",
        data=minutes,
        file_name=f"Meeting_MoM_v{version_index + 1}.txt",
        mime="text/plain",
        width="stretch",
    )
