import sounddevice as sd
import numpy as np
import torch

model, utils = torch.hub.load(
    repo_or_dir='./silero-vad',
    model='silero_vad',
    source="local",
    trust_repo=True
)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

sample_rate = 16000
iterator = VADIterator(model, sampling_rate=sample_rate)

def callback(indata, frames, time, status):
    audio = indata[:, 0].copy()
    speech = iterator(audio, return_seconds=True)
    if speech:
        print(speech)

with sd.InputStream(
    channels=1,
    samplerate=sample_rate,
    blocksize=512,
    callback=callback
):
    print("Listening…")
    while True:
        pass