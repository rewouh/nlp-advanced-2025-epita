import logging
import numpy as np

from enum import Enum
from pathlib import Path
from typing import Optional
import hashlib


try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False

try:
    import sounddevice as sd
    from piper.voice import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False

logger = logging.getLogger(__name__)


class Emotion(Enum):
    """Emotion types for TTS."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"


class TTSEngine:
    """
    TTS Engine supporting emotions and multiple voices.
    
    Uses Coqui XTTS for emotion support, falls back to Piper for speed.
    """
    
    def __init__(
        self,
        use_coqui: bool = True,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",  # Supports emotions
        piper_voice_path: Optional[Path] = None,
    ):
        """
        Initialize TTS engine.
        
        Args:
            use_coqui: Use Coqui XTTS (supports emotions) or Piper (faster)
            model_name: Coqui model name
            piper_voice_path: Path to Piper voice model (fallback)
        """
        if PIPER_AVAILABLE and piper_voice_path and Path(piper_voice_path).exists():
            self.piper_voice = PiperVoice.load(str(piper_voice_path))
        else:
            self.piper_voice = None
        
        self.use_coqui = use_coqui and COQUI_AVAILABLE
        
        if self.use_coqui:
            logger.info(f"Initializing Coqui XTTS with model: {model_name}")
            try:
                import os
                os.environ['COQUI_TOS_AGREED'] = '1'
                
                import torch
                from TTS.tts.configs.xtts_config import XttsConfig
                from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
                from TTS.config.shared_configs import BaseDatasetConfig
                torch.serialization.weights_only = False
                torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])
                
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA GPU is required for XTTS; no CUDA device detected")
                
                device_cap = torch.cuda.get_device_capability()
                
                self.tts = TTS(model_name=model_name)
                self.tts.to("cuda")
                
                self.default_speaker_wav = None
                if 'xtts' in model_name.lower():
                    self.default_speaker_wav = self._create_default_speaker()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Coqui XTTS on GPU: {e}. "
                    "Ensure torch >=2.6.0 with sm_120 support is installed (e.g., nightly cu124)."
                )
        
        if not self.use_coqui:
            raise RuntimeError("Coqui XTTS is required; no fallback enabled.")
    
    def synthesize(
        self,
        text: str,
        emotion: Emotion = Emotion.NEUTRAL,
        speaker_wav: Optional[str] = None,
        language: str = "en",
        speed: float = 1.3,
    ) -> Optional[np.ndarray]:
        """
        Synthesize text to speech with emotion.
        
        Args:
            text: Text to synthesize
            emotion: Emotion to use
            speaker_wav: Optional speaker reference audio (for voice cloning)
            language: Language code
            speed: Speech speed multiplier (0.5-2.0)
            
        Returns:
            Audio array (numpy) or None if failed
        """
        if self.use_coqui:
            return self._synthesize_coqui(text, emotion, speaker_wav, language, speed)
        elif self.piper_voice:
            return self._synthesize_piper(text, speed)
        else:
            logger.error("No TTS backend available")
            return None
    
    def _synthesize_coqui(
        self,
        text: str,
        emotion: Emotion,
        speaker_wav: Optional[str],
        language: str,
        speed: float,
    ) -> Optional[np.ndarray]:
        """
        Synthesize using Coqui XTTS v2 with emotion support.
        
        XTTS v2 supports:
        - Emotion control via emotion parameter
        - Voice cloning via speaker_wav
        - Multi-language support
        """
        try:
            output_path = "/tmp/tts_output.wav"
            
            # Check if model is XTTS v2 (supports emotions)
            model_name = getattr(self.tts, 'model_name', '')
            is_xtts_v2 = 'xtts' in model_name.lower() or 'xtts' in str(type(self.tts)).lower()
            
            if is_xtts_v2:
            # XTTS v2 requires speaker_wav for voice cloning
            # Use default speaker if none provided
                if not speaker_wav:
                    speaker_wav = getattr(self, 'default_speaker_wav', None)
                
                if not speaker_wav:
                    logger.warning("XTTS v2: No speaker_wav, attempting synthesis without it")
                    try:
                        self.tts.tts_to_file(
                            text=text,
                            file_path=output_path,
                            language=language,
                            speed=speed,
                            emotion=emotion.value,
                        )
                    except Exception as e:
                        logger.error(f"XTTS v2 synthesis failed: {e}")
                        return None
                else:
                    # XTTS v2 synthesis with speaker_wav and emotion
                    try:
                        logger.debug(f"Synthesizing with emotion: {emotion.value}, speaker: {speaker_wav[:50] if speaker_wav else 'None'}...")
                        self.tts.tts_to_file(
                            text=text,
                            file_path=output_path,
                            speaker_wav=speaker_wav,
                            language=language,
                            speed=speed,
                            emotion=emotion.value,
                        )
                    except Exception as e:
                        logger.error(f"XTTS v2 synthesis failed: {e}")
                        return None
            else:
                # Other models (no emotion support)
                if speaker_wav:
                    self.tts.tts_to_file(
                        text=text,
                        file_path=output_path,
                        speaker_wav=speaker_wav,
                        language=language,
                        speed=speed,
                    )
                else:
                    self.tts.tts_to_file(
                        text=text,
                        file_path=output_path,
                        language=language,
                        speed=speed,
                    )
            
            # Load audio
            try:
                import soundfile as sf
                audio, _ = sf.read(output_path)
                return audio.astype(np.float32)
            except ImportError:
                # Fallback to scipy.io.wavfile
                try:
                    from scipy.io import wavfile
                    _, audio = wavfile.read(output_path)
                    return audio.astype(np.float32) / 32768.0
                except ImportError:
                    logger.error("Neither soundfile nor scipy available for audio loading")
                    return None
            except Exception as e:
                logger.error(f"Error loading audio: {e}")
                return None
            
        except Exception as e:
            logger.error(f"Coqui synthesis error: {e}")
            return None
    
    def _synthesize_piper(self, text: str, speed: float) -> Optional[np.ndarray]:
        """Synthesize using Piper (fallback, no emotions)."""
        try:
            audio_chunks = list(self.piper_voice.synthesize(text))
            
            if not audio_chunks:
                return None
            
            audio_data = b''.join(chunk.audio_int16_bytes for chunk in audio_chunks)
            sample_rate = audio_chunks[0].sample_rate
            sample_width = audio_chunks[0].sample_width
            channels = audio_chunks[0].sample_channels
            
            dtype = np.int16 if sample_width == 2 else np.int8
            audio_array = np.frombuffer(audio_data, dtype=dtype)
            audio_array = audio_array.reshape(-1, channels)
            
            # Convert to float32 and normalize
            if audio_array.dtype == np.int16:
                audio_array = audio_array.astype(np.float32) / 32768.0
            elif audio_array.dtype == np.int8:
                audio_array = audio_array.astype(np.float32) / 128.0
            else:
                audio_array = audio_array.astype(np.float32)
            
            if speed != 1.0:
                try:
                    from scipy import signal
                    num_samples = int(len(audio_array) / speed)
                    audio_array = signal.resample(audio_array, num_samples)
                except ImportError:
                    logger.warning("scipy not available, speed adjustment skipped")
            
            return audio_array
            
        except Exception as e:
            logger.error(f"Piper synthesis error: {e}")
            return None
    
    def _create_default_speaker(self) -> Optional[str]:
        """
        Create a default speaker_wav file for XTTS v2.
        
        Returns:
            Path to default speaker_wav file or None if failed
        """
        try:
            default_speaker_path = Path("/tmp/xtts_default_speaker.wav")
            
            if default_speaker_path.exists():
                logger.debug(f"Using existing default speaker: {default_speaker_path}")
                return str(default_speaker_path)
            
            if hasattr(self, 'piper_voice') and self.piper_voice:
                audio_chunks = list(self.piper_voice.synthesize("Hello, this is a default voice."))
                if audio_chunks:
                    audio_data = b''.join(chunk.audio_int16_bytes for chunk in audio_chunks)
                    sample_rate = audio_chunks[0].sample_rate
                    
                    import wave
                    with wave.open(str(default_speaker_path), 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(sample_rate)
                        wf.writeframes(audio_data)
                    
                    logger.debug(f"Default speaker created: {default_speaker_path}")
                    return str(default_speaker_path)
            
            logger.warning("Cannot generate default speaker_wav without Piper")
            logger.warning("XTTS v2 will try to work, but may require speaker_wav for each synthesis")
            return None
            
        except Exception as e:
            logger.warning(f"Failed to create default speaker: {e}")
            return None
    
    def get_npc_speaker(self, npc_id: str, voice_gender: Optional[str] = None, traits: list = None) -> Optional[str]:
        base_dir = Path("assets/voices")
        gender_folder = "female" if voice_gender == "female" else "male"
        voice_dir = base_dir / gender_folder
        
        if not voice_dir.exists():
            return self.default_speaker_wav

        available_voices = list(voice_dir.glob("*.wav"))
        available_voices.sort()
        
        if not available_voices:
            return self.default_speaker_wav

        if traits:
            trait_map = {
                "grumpy": ["gruff", "rough", "tough", "angry"],
                "stern": ["gruff", "tough", "deep"],
                "aggressive": ["gruff", "tough"],
                "old": ["old", "elder", "wheezy"],
                "ancient": ["old", "deep"],
                "noble": ["noble", "soft", "posh", "arrogant"],
                "rich": ["noble", "soft"],
                "scholar": ["noble", "soft", "average"],
                "young": ["young", "fast", "excited"],
                "friendly": ["average", "soft", "happy"],
                "mysterious": ["soft", "deep", "whisper"]
            }

            for trait in traits:
                trait_lower = trait.lower()
                if trait_lower in trait_map:
                    target_keywords = trait_map[trait_lower]
                    
                    for voice_file in available_voices:
                        for keyword in target_keywords:
                            if keyword in voice_file.name.lower():
                                logger.info(f"Smart voice match for {npc_id}: {voice_file.name} (Trait: {trait} -> {keyword})")
                                return str(voice_file)

        npc_hash = int(hashlib.md5(npc_id.encode()).hexdigest(), 16)
        voice_index = npc_hash % len(available_voices)
        selected_voice = available_voices[voice_index]
        
        logger.info(f"Random (hashed) voice for {npc_id}: {selected_voice.name}")
        return str(selected_voice)
    
    def synthesize_and_play(
        self,
        text: str,
        emotion: Emotion = Emotion.NEUTRAL,
        speaker_wav: Optional[str] = None,
        language: str = "en",
        speed: float = 1.0,
    ):
        audio = self.synthesize(text, emotion, speaker_wav, language, speed)

        if audio is None:
            return

        try:
            sd.play(audio, samplerate=22050)
            sd.wait()
        except Exception as e:
            logger.error(f"Audio playback error: {e}")

def emotion_from_disposition(disposition: str) -> Emotion:
    """
    Map NPC disposition to emotion.
    
    Args:
        disposition: NPC disposition (hostile, unfriendly, neutral, friendly, trusting)
        
    Returns:
        Emotion enum
    """
    mapping = {
        "hostile": Emotion.ANGRY,
        "unfriendly": Emotion.SAD,
        "neutral": Emotion.NEUTRAL,
        "friendly": Emotion.HAPPY,
        "trusting": Emotion.HAPPY,
    }
    return mapping.get(disposition, Emotion.NEUTRAL)
