# Step 1: Real-Time STT with faster-whisper
# Install first if you haven't:
# pip install faster-whisper sounddevice numpy

import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

# Initialize the model
model_size = "small"  # you can choose tiny, base, small, medium, large
model = WhisperModel(model_size, device="cpu")  # use "cuda" if you have a GPU

# Audio settings
sample_rate = 16000
block_size = 16000  # ~1 sec of audio per block

print("Starting real-time transcription. Speak into your mic...")

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio = indata.flatten().astype(np.float32)
    segments, _ = model.transcribe(audio, beam_size=5)
    for segment in segments:
        print(f"Transcript: {segment.text}")

# Start streaming from mic
with sd.InputStream(channels=1, samplerate=sample_rate, blocksize=block_size, callback=audio_callback):
    print("Press Ctrl+C to stop...")
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStopped real-time transcription.")
