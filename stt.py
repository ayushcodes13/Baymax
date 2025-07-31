import sounddevice as sd
import queue
import sys
import json
from vosk import Model, KaldiRecognizer

q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

model = Model("vosk-model-small-en-in-0.4")
recognizer = KaldiRecognizer(model, 16000)

with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                    channels=1, callback=callback):
    print("🎤 Speak now...")
    while True:
        data = q.get()
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            text = json.loads(result)['text']
            print("📝", text)