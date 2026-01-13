"""
Text-to-Speech (TTS) Engine with emotion and voice cloning support.

This module provides TTS synthesis using Coqui XTTS v2, which supports:
- Emotion control (neutral, happy, sad, angry, fearful, surprised, disgusted)
- Voice cloning via speaker reference audio (speaker_wav)
- Multi-language support
- Prosody and speed control

The engine generates unique voices for each NPC based on their ID and gender,
creating consistent but distinct voice characteristics. It uses a default
speaker generated with Piper TTS as a base, then creates variations for
each NPC using XTTS voice cloning.

Key features:
- GPU-accelerated synthesis (CUDA required)
- Emotion mapping from NPC disposition (hostile -> angry, friendly -> happy)
- Automatic voice gender detection from NPC name/backstory
- Speaker caching for performance
- PyTorch 2.5+ compatibility workarounds

The Emotion enum maps NPC relationship dispositions to TTS emotions,
allowing NPCs to sound appropriately based on their relationship with
the player.

Usage:
    engine = TTSEngine(use_coqui=True)
    audio = engine.synthesize(
        text="Hello, traveler!",
        emotion=Emotion.FRIENDLY,
        speaker_wav="/path/to/speaker.wav"
    )
    engine.play(audio, sample_rate=22050)
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

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
                target_arch = f"sm_{device_cap[0]}{device_cap[1]}"
                if target_arch not in torch.cuda.get_arch_list():
                    raise RuntimeError(
                        f"CUDA arch {target_arch} not supported by this PyTorch build. "
                        "Install torch>=2.6.0 built for your GPU (e.g., nightly cu124 with sm_120 support)."
                    )
                
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
        speed: float = 1.0,
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
            
            # Speed adjustment (simple resampling)
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
                logger.info(f"Using existing default speaker: {default_speaker_path}")
                return str(default_speaker_path)
            
            # Use Piper to generate a default speaker audio (if available)
            if hasattr(self, 'piper_voice') and self.piper_voice:
                logger.info("Generating default speaker using Piper...")
                audio_chunks = list(self.piper_voice.synthesize("Hello, this is a default voice."))
                if audio_chunks:
                    audio_data = b''.join(chunk.audio_int16_bytes for chunk in audio_chunks)
                    sample_rate = audio_chunks[0].sample_rate
                    
                    # Save as WAV file
                    import wave
                    with wave.open(str(default_speaker_path), 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(sample_rate)
                        wf.writeframes(audio_data)
                    
                    logger.info(f"Default speaker created: {default_speaker_path}")
                    return str(default_speaker_path)
            
            # Cannot use XTTS to generate speaker (circular dependency - XTTS needs speaker_wav)
            logger.warning("Cannot generate default speaker_wav without Piper")
            logger.warning("XTTS v2 will try to work, but may require speaker_wav for each synthesis")
            return None
            
        except Exception as e:
            logger.warning(f"Failed to create default speaker: {e}")
            return None
    
    def get_npc_speaker(self, npc_id: str, voice_gender: Optional[str] = None) -> Optional[str]:
        """
        Generate or retrieve a unique speaker voice for an NPC.
        
        Uses a hash of the NPC ID and gender to create consistent but unique voices.
        
        Args:
            npc_id: NPC identifier
            voice_gender: Optional voice gender ('male', 'female', or None for auto)
            
        Returns:
            Path to speaker_wav file, or None if generation fails
        """
        import hashlib
        from pathlib import Path
        
        # Create a hash-based filename for this NPC (including gender)
        hash_input = f"{npc_id}_{voice_gender or 'auto'}"
        npc_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        speaker_path = Path(f"/tmp/xtts_speaker_{npc_hash}.wav")
        
        # Return existing speaker if available
        if speaker_path.exists():
            return str(speaker_path)
        
        # Generate new speaker using default speaker as base
        if not self.default_speaker_wav:
            logger.warning(f"Cannot generate speaker for {npc_id}: no default speaker available")
            return None
        
        try:
            # Generate voice variation using XTTS
            # Adjust text based on gender for more appropriate voice
            if voice_gender == "female":
                variation_text = f"Hello, my name is {npc_id}. I am speaking to you now."
            elif voice_gender == "male":
                variation_text = f"Hello, I am {npc_id}. This is my voice speaking."
            else:
                variation_text = f"Hello, this is {npc_id} speaking."
            
            # Generate voice variation
            temp_output = Path(f"/tmp/xtts_temp_{npc_hash}.wav")
            self.tts.tts_to_file(
                text=variation_text,
                file_path=str(temp_output),
                speaker_wav=self.default_speaker_wav,
                language="en",
                emotion=Emotion.NEUTRAL.value,
            )
            
            # Copy to final location
            import shutil
            shutil.copy(str(temp_output), str(speaker_path))
            temp_output.unlink()
            
            logger.debug(f"Generated unique speaker for {npc_id}: {speaker_path}")
            return str(speaker_path)
            
        except Exception as e:
            logger.warning(f"Failed to generate speaker for {npc_id}: {e}")
            # Fallback to default speaker
            return self.default_speaker_wav
    
    def play(self, audio: np.ndarray, sample_rate: int = 22050):
        """
        Play audio.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
        """
        try:
            sd.play(audio, samplerate=sample_rate)
            sd.wait()
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
    
    def synthesize_and_play(
        self,
        text: str,
        emotion: Emotion = Emotion.NEUTRAL,
        speaker_wav: Optional[str] = None,
        language: str = "en",
        speed: float = 1.0,
    ):
        audio = self.synthesize(text, emotion, speaker_wav, language, speed)
        if audio is not None:
            self.play(audio)


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


def create_tts_engine(
    use_coqui: bool = True,
    piper_fallback_path: Optional[Path] = None,
) -> TTSEngine:
    """
    Factory function to create TTS engine.
    
    Args:
        use_coqui: Use Coqui XTTS if available
        piper_fallback_path: Path to Piper model for fallback
        
    Returns:
        TTSEngine instance
    """
    return TTSEngine(
        use_coqui=use_coqui,
        piper_voice_path=piper_fallback_path,
    )

