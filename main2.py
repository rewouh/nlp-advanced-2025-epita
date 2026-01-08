import os
import tempfile
import queue
import threading
import time
import json
import warnings
import audioop
import pyaudio
import wave

import experiments.stt as stt
from langchain_community.chat_models import ChatOllama
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import LLMChain

warnings.filterwarnings(
    "ignore",
    message="FP16 is not supported on CPU; using FP32 instead"
)

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

CHUNK_SECONDS = 2.0
OVERLAP_SECONDS = 0.5

SILENCE_RMS_THRESHOLD = 300
SILENCE_TRIGGER_SECONDS = 0.4

def is_silence(raw_audio: bytes, threshold=SILENCE_RMS_THRESHOLD):
    return audioop.rms(raw_audio, 2) < threshold


class MiniBrain:
    def __init__(self, model="qwen2.5:3b"):
        self.llm = ChatOllama(
            model=model,
            temperature=0.2,
            max_tokens=200,
        )

        template = """You are a transcription assistant.
You receive a list of spoken sentences from a conversation.
Your task:
- Merge them into a single coherent sentence.
- Correct obvious transcription errors.
- Remove duplicates (repeated words).
- Keep the meaning intact.
- Do not develop idea, do not try to guess and imagine the rest of the conversation.

Previous sentence (for context, may be empty): {context}
New sentences: {sentences}

Output:"""

        self.prompt = PromptTemplate(
            input_variables=["context", "sentences"],
            template=template,
        )

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def think(self, sentences: str, context: list[str]):
        raw = self.chain.run(
            context=json.dumps(context, ensure_ascii=False),
            sentences=sentences,
        )

        return raw.strip()


audio_queue = queue.Queue()

mini_buffer: list[str] = []
full_transcript: list[str] = []

brain = MiniBrain()


def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    frames_buffer = []
    silence_duration = 0.0

    overlap_frames = int(OVERLAP_SECONDS * RATE)
    chunk_frames = int(CHUNK_SECONDS * RATE)

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames_buffer.append(data)

            if is_silence(data):
                silence_duration += CHUNK / RATE
            else:
                silence_duration = 0.0

            total_frames = len(frames_buffer) * CHUNK

            if total_frames >= chunk_frames:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                wf = wave.open(tmp.name, "wb")
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b"".join(frames_buffer))
                wf.close()

                audio_queue.put(("audio", tmp.name))

                keep = int(overlap_frames / CHUNK)
                frames_buffer = frames_buffer[-keep:] if keep > 0 else []

            if silence_duration >= SILENCE_TRIGGER_SECONDS:
                print("Silence")
                audio_queue.put(("silence", None))
                silence_duration = 0.0

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def flush_buffer():
    global mini_buffer, full_transcript

    if not mini_buffer:
        return

    raw_text = " ".join(mini_buffer)
    cleaned = brain.think(raw_text, context=full_transcript)

    print("Summarized:", cleaned)
    full_transcript.append(cleaned)
    print("Full transcription")
    for f in full_transcript:
        print("-", f)

    mini_buffer.clear()


def process_audio():
    while True:
        item = audio_queue.get()

        if item is None:
            flush_buffer()
            break

        kind, payload = item

        if kind == "audio":
            text = stt.speech_to_text(payload)
            os.remove(payload)

            if text and text.strip():
                mini_buffer.append(text.strip())
                print("Raw:", text.strip())

            if len(mini_buffer) >= 5:
                flush_buffer()

        elif kind == "silence":
            flush_buffer()


def main():
    record_thread = threading.Thread(target=record_audio, daemon=True)
    process_thread = threading.Thread(target=process_audio, daemon=True)

    record_thread.start()
    process_thread.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        audio_queue.put(None)
        record_thread.join()
        process_thread.join()


if __name__ == "__main__":
    main()
