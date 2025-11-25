import os
import experiments.record as record
import experiments.stt as stt
import experiments.tts as tts
from experiments.brain import NPCBrain

def main():
    # Config du NPC
    npc = NPCBrain(
        char_name="Joe",
        context="You are a grumpy tavern keeper inside a fantasy RPG. You don't like strangers.",
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
            response = npc.think(user_text)
            print(f"Joe: {response}")

            # 4. Speak
            tts.text_to_speech(response)
            
            # Nettoyage fichier temp
            os.remove(audio_file)

    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()