import argparse
import subprocess
import tempfile
import os

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
import numpy as np
from pywhispercpp.model import Model as WhisperModel

from agent_loop import run_task
from voice_loop import transcribe, speak, WHISPER_MODEL_SIZE

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Loaded once at startup, reused across requests.
whisper_model = WhisperModel(WHISPER_MODEL_SIZE)

# Set from --app at startup; which Mac application the agent controls.
TARGET_APP = "Notes"


@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html") as f:
        return f.read()


@app.post("/command/text")
async def command_text(text: str = Form(...)):
    """Text command from the phone -> straight into the agent loop."""
    result = await run_in_threadpool(run_task, TARGET_APP, text)
    speak("Done")
    return JSONResponse({"heard": text, "result": str(result)})


@app.post("/command/voice")
async def command_voice(audio: UploadFile):
    """Voice command from the phone: webm/opus upload -> wav -> whisper ->
    agent loop, reusing the exact same transcribe() used by the local
    wake-word flow in voice_loop.py."""
    raw_bytes = await audio.read()

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as raw_f:
        raw_f.write(raw_bytes)
        raw_path = raw_f.name

    wav_path = raw_path.replace(".webm", ".wav")
    try:
        # Convert to 16kHz mono PCM WAV, which is what transcribe() expects.
        subprocess.run(
            ["ffmpeg", "-y", "-i", raw_path, "-ar", "16000", "-ac", "1", wav_path],
            check=True,
            capture_output=True,
        )

        # Read raw PCM samples out of the wav file without extra deps.
        import wave
        with wave.open(wav_path, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
        audio_int16 = np.frombuffer(frames, dtype=np.int16)

        heard_text = await run_in_threadpool(transcribe, whisper_model, audio_int16)
    finally:
        os.remove(raw_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)

    if not heard_text.strip():
        return JSONResponse({"heard": "", "result": "Didn't catch that -- try again."})

    result = await run_in_threadpool(run_task, TARGET_APP, heard_text)
    speak("Done")
    return JSONResponse({"heard": heard_text, "result": str(result)})


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--app", default="Notes", help="Mac app for the agent to control")
    parser.add_argument("--port", type=int, default=8420)
    args = parser.parse_args()

    TARGET_APP = args.app

    uvicorn.run(app, host="0.0.0.0", port=args.port)