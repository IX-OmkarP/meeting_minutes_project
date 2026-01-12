import streamlit as st
from openai import OpenAI
import tempfile
import os
from dotenv import load_dotenv
from datetime import datetime

# ---------------- Load .env ----------------
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    st.error("OPENAI_API_KEY not found in .env file. Please check your .env file.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="AI Meeting Minutes Generator", layout="centered")
st.title("🎙️ AI Meeting Minutes Generator")
st.write("Upload any meeting recording and get professional Meeting Minutes instantly.")

# ---------------- Transcription ----------------
def transcribe_audio(file_path):
    """Transcribe audio using OpenAI Whisper API."""
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcript.text

# ---------------- Corporate MoM Generation ----------------
def generate_minutes(transcript, meeting_title, location, date, time, attendees):
    """
    Generate professional corporate Meeting Minutes with Agenda, Decisions, and Action Items.
    """
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
        model="gpt-4o-mini",
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

        # Transcribe audio directly (OpenAI Whisper supports multiple formats)
        st.info("Transcribing audio using OpenAI API...")
        try:
            transcript = transcribe_audio(input_path)
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            st.stop()

        st.subheader("📄 Transcript")
        st.text_area("", transcript, height=200)

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
        st.text_area("", minutes, height=300)

        # ---------------- Download Button ----------------
        st.download_button(
            label="📥 Download MoM as TXT",
            data=minutes,
            file_name=f"{meeting_title.replace(' ', '_')}_MoM.txt",
            mime="text/plain"
        )
