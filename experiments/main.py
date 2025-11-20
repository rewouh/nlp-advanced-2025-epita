import experiments.record as record
import experiments.stt as stt
import experiments.tts as tts

if __name__ == "__main__":
    audio_file = record.record_audio(duration=5)
    model = stt.whisper.load_model("base")
    result = model.transcribe(audio_file)
    print("Transcription:", result["text"])
    tts.text_to_speech(result["text"])