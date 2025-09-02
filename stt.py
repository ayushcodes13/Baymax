import os
import speech_recognition as sr
import whisper

# ------------------ CONFIG ------------------
# Path to local Whisper model (download and put in cache or any folder)
base_model_path = os.path.expanduser('~/.cache/whisper/base.pt')
stt_model = whisper.load_model(base_model_path)

recognizer = sr.Recognizer()
mic = sr.Microphone()

# ------------------ FUNCTIONS ------------------
def listen_and_transcribe():
    """Listen to microphone and return transcription"""
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        audio = recognizer.listen(source)
        # Save to temp file
        with open("temp_command.wav", "wb") as f:
            f.write(audio.get_wav_data())
    # Transcribe using local Whisper
    result = stt_model.transcribe("temp_command.wav")
    text = result['text'] if result and result['text'] else ""
    print("Transcript:", text)
    return text

# ------------------ MAIN LOOP ------------------
if __name__ == "__main__":
    print("STT module started. Press Ctrl+C to stop.")
    try:
        while True:
            listen_and_transcribe()
    except KeyboardInterrupt:
        print("\nSTT module stopped.")
