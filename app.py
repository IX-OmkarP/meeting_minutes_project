import streamlit as st
from groq import Groq
import tempfile
import os
from dotenv import load_dotenv
from datetime import datetime
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

from transcription_utils import get_chunk_windows

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


def transcribe_audio(file_path):
    """Downsample audio, then transcribe. Chunk if still over the size limit."""
    audio = AudioSegment.from_file(file_path).set_frame_rate(16000).set_channels(1)

    original_size_bytes = os.path.getsize(file_path)
    tmpdir = os.path.dirname(file_path)
    compressed_path = os.path.join(tmpdir, "compressed.mp3")
    audio.export(compressed_path, format="mp3", bitrate=f"{TARGET_BITRATE_KBPS}k")
    compressed_size_bytes = os.path.getsize(compressed_path)

    st.info(
        f"Upload size: {original_size_bytes / (1024 * 1024):.2f} MB → compressed size: {compressed_size_bytes / (1024 * 1024):.2f} MB"
    )

    if compressed_size_bytes <= MAX_UPLOAD_BYTES:
        return _transcribe_file(compressed_path)

    windows = get_chunk_windows(
        total_ms=len(audio),
        max_bytes=MAX_UPLOAD_BYTES,
        max_chunk_ms=MAX_CHUNK_MS,
        bitrate_kbps=TARGET_BITRATE_KBPS,
        min_chunk_ms=MIN_CHUNK_MS,
    )
    st.info(f"Audio was split into {len(windows)} chunk(s) for transcription.")

    if not windows:
        raise ValueError("Unable to prepare audio for transcription")

    parts = []
    for i, (start_ms, end_ms) in enumerate(windows):
        chunk = audio[start_ms:end_ms]
        chunk_path = os.path.join(tmpdir, f"chunk_{i}.mp3")
        chunk.export(chunk_path, format="mp3", bitrate=f"{TARGET_BITRATE_KBPS}k")
        parts.append(_transcribe_file(chunk_path))

    return " ".join(parts)

# ---------------- Corporate MoM Generation ----------------
def generate_minutes(transcript, meeting_title, location, date, time, attendees):
    prompt = f"""
You are a professional assistant. Convert the following transcript into a corporate-style Meeting Minutes exactly in this format:

Meeting Minutes – {meeting_title}

Location:           {location}
Date:               {date}
Time:               {time}

Attendance
{attendees}

Agenda
{transcript}

Instructions:
- Clearly separate Agenda, Decisions, and Action Items.
- Agenda should summarize discussion points in full sentences.
- Decisions and Action Items must be in bullet points.
- Do NOT include 'Next Steps', 'Adjournment', 'Next Meeting', 'Minutes Prepared by', or any placeholders.
- Maintain professional corporate formatting.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

# ---------------- User Inputs ----------------
uploaded_file = st.file_uploader(
    "Upload meeting recording",
    type=["mp3", "wav", "m4a", "ogg", "mp4"]
)

meeting_title = st.text_input("Meeting Title", "CorePoint Client Call")
location = st.text_input("Location", "Pune")
date = st.date_input("Date", datetime.now())
time = st.time_input("Time", datetime.now())
attendees = st.text_area("Attendance (comma-separated)", "Lisa Heitrich, Omkar Pawar, Chinmay Dole")

# ---------------- Process Upload ----------------
if uploaded_file:
    st.audio(uploaded_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, uploaded_file.name)

        # Save uploaded file temporarily
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        # Transcribe directly
        st.info("Transcribing audio using Groq Whisper...")
        try:
            transcript = transcribe_audio(input_path)
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            st.stop()

        st.subheader("📄 Transcript")
        st.text_area(
            label="Transcript Output",
            value=transcript,
            height=220,
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Generate corporate Meeting Minutes
        st.info("Generating professional Meeting Minutes...")
        try:
            minutes = generate_minutes(
                transcript,
                meeting_title=meeting_title,
                location=location,
                date=date.strftime("%d-%m-%Y"),
                time=time.strftime("%I:%M %p"),
                attendees=attendees
            )
        except Exception as e:
            st.error(f"Meeting minutes generation failed: {e}")
            st.stop()

        st.subheader("📝 Meeting Minutes")
        st.text_area(
            label="Meeting Minutes Output",
            value=minutes,
            height=320,
            label_visibility="collapsed"
        )

        # ---------------- Download Button ----------------
        st.download_button(
            label="📥 Download MoM as TXT",
            data=minutes,
            file_name=f"{meeting_title.replace(' ', '_')}_MoM.txt",
            mime="text/plain"
        )
