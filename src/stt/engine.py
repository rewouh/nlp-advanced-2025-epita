import logging
import numpy as np

from pathlib import Path
from typing import Optional, List, Union
from src.rag.models import WorldLore

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    import whisper

logger = logging.getLogger(__name__)


class HotwordExtractor:
    """Extract hotwords from world lore for improved transcription accuracy."""
    
    @staticmethod
    def extract_from_world(world_lore: WorldLore) -> List[str]:
        """
        Extract all relevant proper nouns from world lore.
        
        Args:
            world_lore: WorldLore instance containing game data
            
        Returns:
            List of hotwords (names, locations, items, etc.)
        """
        hotwords = set()
        
        for npc in world_lore.npcs:
            hotwords.add(npc.name)
            if npc.id:
                hotwords.add(npc.id.replace("_", " ").title())
        
        for location in world_lore.locations:
            hotwords.add(location.name)
            if location.id:
                hotwords.add(location.id.replace("_", " ").title())
        
        for item in world_lore.items:
            hotwords.add(item.name)
        
        for faction in world_lore.factions:
            hotwords.add(faction.name)
        
        for event in world_lore.history:
            hotwords.add(event.event)
        
        for quest in world_lore.quests:
            pass
        
        rpg_vocab = [
            "drink", "drinks", "ale", "beer", "tavern", "inn", "quest", "quests",
            "gold", "silver", "coin", "coins", "sword", "shield", "armor",
            "magic", "spell", "spells", "potion", "potions", "dungeon", "dragon"
        ]
        hotwords.update(rpg_vocab)
        
        cleaned = []
        for word in hotwords:
            if word and len(word) > 2 and word.lower() not in ["the", "and", "for"]:
                cleaned.append(word)
        
        return sorted(cleaned)
    
    @staticmethod
    def format_for_whisper(hotwords: List[str]) -> str:
        return ", ".join(hotwords[:50])


class STTEngine:
    """
    Singleton Speech-to-Text engine using faster-whisper.
    
    Optimized for low latency with GPU support and custom vocabulary.
    """
    
    _instance: Optional['STTEngine'] = None
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
        hotwords: Optional[List[str]] = None,
    ):
        """
        Initialize STT engine.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (cuda, cpu, auto)
            compute_type: Computation type (float16, int8, auto)
            hotwords: Optional list of custom vocabulary words
        """
        if not FASTER_WHISPER_AVAILABLE:
            logger.warning(
                "faster-whisper not available, falling back to standard whisper. "
                "Install with: pip install faster-whisper"
            )
            self.use_faster_whisper = False
            self.model = whisper.load_model(model_size)
        else:
            self.use_faster_whisper = True
            
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if compute_type == "auto":
                compute_type = "float16" if device == "cuda" else "int8"
            
            logger.info(
                f"Initializing faster-whisper with model={model_size}, "
                f"device={device}, compute_type={compute_type}"
            )
            
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
            )
        
        self.model_size = model_size
        self.device = device
        self.hotwords = hotwords or []
        self.hotword_prompt = HotwordExtractor.format_for_whisper(self.hotwords)
        
        logger.info(f"STT Engine initialized with {len(self.hotwords)} hotwords")
    
    @classmethod
    def get_instance(
        cls,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
        hotwords: Optional[List[str]] = None,
    ) -> 'STTEngine':
        """
        Get or create singleton instance.
        
        Args:
            model_size: Whisper model size
            device: Device to run on
            compute_type: Computation type
            hotwords: Custom vocabulary
            
        Returns:
            STTEngine instance
        """
        if cls._instance is None:
            cls._instance = cls(model_size, device, compute_type, hotwords)

        return cls._instance
    
    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray],
        language: str = "en",
        temperature: float = 0.0,
        vad_filter: bool = True,
    ) -> str:
        """
        Args:
            audio: Audio file path or numpy array (16kHz)
            language: Language code (en, fr, etc.)
            temperature: Sampling temperature (0 = deterministic)
            vad_filter: Apply voice activity detection filter
            
        Returns:
            Transcribed text
        """
        if self.use_faster_whisper:
            return self._transcribe_faster_whisper(audio, language, temperature, vad_filter)
        else:
            return self._transcribe_whisper(audio, language, temperature)
    
    def _transcribe_faster_whisper(
        self,
        audio: Union[str, Path, np.ndarray],
        language: str,
        temperature: float,
        vad_filter: bool,
    ) -> str:
        initial_prompt = self.hotword_prompt if self.hotword_prompt else None
        segments, info = self.model.transcribe(
            audio,
            language=language,
            temperature=temperature,
            vad_filter=vad_filter,
            initial_prompt=initial_prompt,
            beam_size=5,
            best_of=5,
            patience=1.0,
            length_penalty=1.0,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=True,
            word_timestamps=False,
        )
        
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())
        
        return " ".join(text_parts).strip()
    
    def _transcribe_whisper(
        self,
        audio: Union[str, Path, np.ndarray],
        language: str,
        temperature: float,
    ) -> str:
        result = self.model.transcribe(
            str(audio),
            language=language,
            temperature=temperature,
            initial_prompt=self.hotword_prompt,
        )
        return result["text"].strip()
    
    def set_hotwords(self, hotwords: List[str]):
        """
        Args:
            hotwords: New list of custom vocabulary
        """
        self.hotwords = hotwords
        self.hotword_prompt = HotwordExtractor.format_for_whisper(hotwords)
        logger.info(f"Updated hotwords: {len(hotwords)} words")
    
    def add_hotwords(self, new_hotwords: List[str]):
        """
        Args:
            new_hotwords: Additional custom vocabulary
        """
        self.hotwords.extend(new_hotwords)
        self.hotwords = list(set(self.hotwords))  # Remove duplicates
        self.hotword_prompt = HotwordExtractor.format_for_whisper(self.hotwords)
        logger.info(f"Added hotwords, total: {len(self.hotwords)} words")


def create_stt_engine(
    world_lore: Optional[WorldLore] = None,
    model_size: str = "base",
    device: str = "auto",
    compute_type: str = "auto",
) -> STTEngine:
    """
    Factory function to create STT engine with world lore hotwords.
    
    Args:
        world_lore: Optional WorldLore to extract hotwords from
        model_size: Whisper model size
        device: Device to run on
        compute_type: Computation type
        
    Returns:
        Configured STTEngine instance
    """
    hotwords = []
    if world_lore:
        hotwords = HotwordExtractor.extract_from_world(world_lore)
        logger.debug(f"Extracted {len(hotwords)} hotwords from world lore")
    
    return STTEngine.get_instance(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        hotwords=hotwords,
    )

