import experiments.record as record
import experiments.stt as stt
import experiments.tts as tts

if __name__ == "__main__":
    audio_file = record.record_audio(duration=5)
    result = stt.speech_to_text(audio_file)
    print("Transcription:", result)
    tts.text_to_speech(result)