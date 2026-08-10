import argparse
import subprocess
import time
import numpy as np
import pyaudio

from openwakeword.model import Model as WakeWordModel
from openwakeword.utils import download_models
from pywhispercpp.model import Model as WhisperModel

from agent_loop import run_task

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280          
WAKE_THRESHOLD = 0.5
RECORD_SECONDS = 5         
WHISPER_MODEL_SIZE = "base.en"  


def speak(text):
    print(f"[speaking] {text}")
    subprocess.run(["say", text])


def open_mic_stream():
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )
    return pa, stream


def record_audio(stream, seconds=RECORD_SECONDS):
    """Record raw audio for a fixed duration after the wake word fires."""
    frames = []
    num_chunks = int(SAMPLE_RATE / CHUNK_SIZE * seconds)
    for _ in range(num_chunks):
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        frames.append(np.frombuffer(data, dtype=np.int16))
    return np.concatenate(frames)


def transcribe(whisper_model, audio_int16):
   
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    segments = whisper_model.transcribe(audio_float32)
    text = " ".join(seg.text for seg in segments).strip()

    stripped = text.strip()
    if stripped.startswith(("(", "[")) and stripped.endswith((")", "]")):
        print(f"Discarding likely hallucinated non-speech tag: {text!r}")
        return ""

    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", required=True, help="App the agent should control")
    parser.add_argument("--wake-threshold", type=float, default=WAKE_THRESHOLD)
    args = parser.parse_args()

    print("Loading wake word model (hey jarvis)...")
    download_models()  # no-op if already downloaded; fetches missing model files
    wake_model = WakeWordModel(wakeword_models=["hey_jarvis_v0.1"], inference_framework="onnx")

    print(f"Loading whisper.cpp model ({WHISPER_MODEL_SIZE}) — first run downloads it...")
    whisper_model = WhisperModel(WHISPER_MODEL_SIZE)

    pa, stream = open_mic_stream()

    print(f'\nListening for "hey jarvis"... (Ctrl+C to stop)\n')

    try:
        while True:
            chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_chunk = np.frombuffer(chunk, dtype=np.int16)

            prediction = wake_model.predict(audio_chunk)
            score = prediction.get("hey_jarvis_v0.1", 0.0)

            if score > args.wake_threshold:
                print(f"\nWake word detected (score={score:.2f})")
                speak("Listening")
                time.sleep(0.3)  # brief reaction-time buffer before recording starts

                audio = record_audio(stream)
                print("Transcribing...")
                goal_text = transcribe(whisper_model, audio)

                if not goal_text:
                    print("Heard nothing usable — going back to listening.")
                    speak("Sorry, I didn't catch that")
                    wake_model.reset()
                    continue

                print(f'Heard: "{goal_text}"')
                speak(f"Got it: {goal_text}")

                run_task(args.app, goal_text)

                speak("Done")
                wake_model.reset() 
                print(f'\nListening for "hey jarvis"...\n')

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


if __name__ == "__main__":
    main()