from piper.voice import PiperVoice
import sounddevice as sd
import numpy as np
import wave

def text_to_speech(text):
    voice = PiperVoice.load("./experiments/en_US-joe-medium.onnx")
    audio_chunks = list(voice.synthesize(text))
    
    for chunk in audio_chunks:
        audio_data = b''.join(chunk.audio_int16_bytes for chunk in audio_chunks)
        sample_rate = chunk.sample_rate
        sample_width = chunk.sample_width
        channels = chunk.sample_channels

        dtype = np.int16 if sample_width == 2 else np.int8
        audio_array = np.frombuffer(audio_data, dtype=dtype)
        audio_array = audio_array.reshape(-1, channels)

        sd.play(audio_array, samplerate=sample_rate)
        sd.wait()
