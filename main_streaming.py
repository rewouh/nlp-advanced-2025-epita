import os
import experiments.record as record
import experiments.stt as stt
import experiments.tts as tts
from experiments.brain import NPCBrain
import threading

from collections import deque

def main():
    # Config du NPC
    npc = NPCBrain(
        char_name="Joe",
        context="You are a friendly and helpful tavern keeper inside a fantasy RPG. You love chatting with travelers and answering their questions.",
        model="qwen2.5:3b"
    )

    print(f"--- System Ready (Ctrl+C to stop) ---")

    try:
        while True:
            # 1. Record
            print("\nListening...")
            audio_file = record.record_audio(duration=4)
            
            # 2. Transcribe
            user_text = stt.speech_to_text(audio_file)
            
            # Filtre silence/bruit court
            if not user_text.strip() or len(user_text) < 2:
                continue
                
            print(f"User: {user_text}")

            # 3. Think (LangChain)

            n_chunks_tts = 15
            chunks = ''

            for chunk in npc.think(user_text):
                print(f'Chunk for TTS: \"{chunk}\"')
                print(f'Accumulated chunks: \"{chunks}\"')
                chunks += chunk
                
                if len(chunks) >= n_chunks_tts:
                    # tts in a separate thread to avoid blocking
                    threading.Thread(target=tts.text_to_speech, args=(chunks,)).start()
                    chunks = ''

            # Send any remaining chunks
            if chunks.strip():
                tts.text_to_speech(chunks)

            # 4. Speak
            # tts.text_to_speech(response)
            
            # Nettoyage fichier temp
            os.remove(audio_file)

    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()