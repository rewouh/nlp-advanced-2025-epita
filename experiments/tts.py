from piper.voice import PiperVoice
import sounddevice as sd
import numpy as np
import wave
import threading

sd.default.device = (None, 0)  # (input, output)

voice = PiperVoice.load("./experiments/en_US-joe-medium.onnx")
audio_lock = threading.Lock()

def text_to_speech(text):
    audio_chunks = list(voice.synthesize(text))
    
    chunk = audio_chunks[0]

    audio_data = b''.join(chunk.audio_int16_bytes for chunk in audio_chunks)
    sample_rate = chunk.sample_rate
    sample_width = chunk.sample_width
    channels = chunk.sample_channels

    dtype = np.int16 if sample_width == 2 else np.int8
    audio_array = np.frombuffer(audio_data, dtype=dtype)
    audio_array = audio_array.reshape(-1, channels)

    with audio_lock:
        sd.play(audio_array, samplerate=sample_rate)
        sd.wait()