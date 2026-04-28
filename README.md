# AI Meeting Minutes Generator

Streamlit app that transcribes meeting recordings and generates professional Meeting Minutes (MoM) using Groq AI.

## How It Works

1. Upload audio recording (mp3, wav, m4a, ogg, mp4)
2. Groq Whisper transcribes audio → text
3. LLaMA 3.3 70B formats transcript → corporate MoM
4. Download MoM as `.txt`

## Prerequisites

- Python 3.13
- [Groq API key](https://console.groq.com/keys)
- ffmpeg installed on system

### Install ffmpeg

**Windows:**
```bash
winget install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

## Setup

**1. Clone repo**
```bash
git clone <repo-url>
cd CP_MInute
```

**2. Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure API key**

Create `.env` file in project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your key at: https://console.groq.com/keys

## Run

```bash
streamlit run app.py
```

App opens at `http://localhost:8501`

## Usage

1. Fill in meeting details (title, location, date, time, attendees)
2. Upload audio file
3. Wait for transcription + MoM generation
4. Download MoM as TXT

## Project Structure

```
CP_MInute/
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
├── runtime.txt         # Python version (3.13)
├── .env                # API keys (DO NOT commit)
└── .gitignore
```

## Models Used

| Task | Model |
|------|-------|
| Transcription | `whisper-large-v3` (Groq) |
| MoM Generation | `llama-3.3-70b-versatile` (Groq) |

## Supported Audio Formats

`mp3` · `wav` · `m4a` · `ogg` · `mp4`

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key for Whisper + LLaMA |
