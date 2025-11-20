import whisper

def speech_to_text(audio_path):
    model = whisper.load_model("medium")
    result = model.transcribe(
        audio_path,
        language="en",
        task="transcribe",
        temperature=0,
        no_speech_threshold=0.5,
        logprob_threshold=-1.0
    )
    return result["text"]