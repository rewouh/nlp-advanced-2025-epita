#!/usr/bin/env python3
import logging
import sys
import signal
import argparse
import os
import subprocess
import warnings

from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('overhearing.log'),
    ]
)

logging.getLogger("jieba").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import OverhearingPipeline
from src.evaluation.evaluate import evaluate

def setup_tts():
    """
    Initialize TTS engine with emotion and voice cloning support.
    
    Returns:
        TTS callback function that accepts (text, emotion, npc_id, voice_gender)
    """
    try:
        from src.tts import TTSEngine, Emotion, emotion_from_disposition
        
        piper_fallback = Path(__file__).parent.parent / "experiments" / "en_US-joe-medium.onnx"
        
        tts_engine = TTSEngine(
            use_coqui=True,
            piper_voice_path=piper_fallback if piper_fallback.exists() else None,
        )
        
        npc_speakers: dict[str, str] = {}
        
        def tts_callback(text: str, emotion: str = None, npc_id: str = None, voice_gender: str = None):
            """
            Synthesize and play text with optional emotion and NPC-specific voice.
            
            Args:
                text: Text to synthesize
                emotion: Emotion/disposition (hostile, unfriendly, neutral, friendly, trusting)
                npc_id: Optional NPC ID for voice cloning
                voice_gender: Optional voice gender ('male', 'female', or None for auto)
            """
            try:
                if emotion:
                    emotion_enum = emotion_from_disposition(emotion)
                else:
                    emotion_enum = Emotion.NEUTRAL
                
                speaker_wav = None
                if npc_id:
                    cache_key = f"{npc_id}_{voice_gender or 'auto'}"
                    if cache_key not in npc_speakers:
                        speaker_wav = tts_engine.get_npc_speaker(npc_id, voice_gender=voice_gender)
                        if speaker_wav:
                            npc_speakers[cache_key] = speaker_wav
                    else:
                        speaker_wav = npc_speakers[cache_key]
                
                tts_engine.synthesize_and_play(
                    text=text,
                    emotion=emotion_enum,
                    speaker_wav=speaker_wav,
                    language="en",
                )
                
            except Exception as e:
                logger.error(f"TTS error: {e}")
        
        return tts_callback
        
    except ImportError as e:
        logger.warning(f"TTS not available: {e}, text output only")
        return None


def run_pipeline():
    """Run the main overhearing pipeline."""
    logger.info("=" * 60)
    logger.info("OVERHEARING AGENTS - NPC Conversation System")
    logger.info("=" * 60)

    ui_process = subprocess.Popen(
        [sys.executable, "ui/app.py"],
        env=os.environ,
        stderr=sys.stderr
    )
    
    project_root = Path(__file__).parent.parent
    world_path = project_root / "lore" / "world.yaml"
    session_path = project_root / "lore" / "session.yaml"
    
    if not world_path.exists():
        logger.error(f"World file not found: {world_path}")
        return 1
    
    tts_callback = setup_tts()
    
    try:
        pipeline = OverhearingPipeline(
            world_path=world_path,
            session_path=session_path if session_path.exists() else None,
            model="qwen2.5:7b",
            stt_model="base",
            tts_callback=tts_callback,
        )
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
        return 1
    
    def signal_handler(sig, frame):
        logger.info("\nShutting down gracefully...")
        pipeline.stop()

        if ui_process.poll() is not None:
            sys.exit(0)

        os.killpg(ui_process.pid, signal.SIGKILL)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("=" * 60)
    logger.info("LISTENING FOR CONVERSATIONS...")
    logger.info("=" * 60)
    
    pipeline.start()
    
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    
    pipeline.stop()
    
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval",
        action="store_true",
    )
    
    args = parser.parse_args()
    
    if args.eval:
        os.environ["DISABLE_UI"] = "1"
        logger.info("Running evaluation tests...")
        return evaluate()
    else:
        return run_pipeline()


if __name__ == "__main__":
    sys.exit(main())
