import logging
import queue
import threading
import time
import numpy as np
import sounddevice as sd
import torch

from typing import Optional, Callable
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class AudioChunk:
    audio: np.ndarray
    sample_rate: int
    timestamp: float
    is_speech: bool

class AudioBuffer:
    """
    Thread-safe circular buffer for audio chunks.
    """    
    def __init__(self, max_duration_seconds: float = 10.0, sample_rate: int = 16000):
        """
        Args:
            max_duration_seconds: Maximum duration to keep in buffer
            sample_rate: Audio sample rate
        """
        self.max_samples = int(max_duration_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.buffer = deque(maxlen=self.max_samples)
        self._lock = threading.Lock()
    
    def append(self, audio: np.ndarray):
        """
        Args:
            audio: Audio samples to add
        """
        with self._lock:
            self.buffer.extend(audio)
    
    def get_all(self) -> np.ndarray:
        """
        Returns:
            Concatenated audio array
        """
        with self._lock:
            if not self.buffer:
                return np.array([], dtype=np.float32)
            return np.array(list(self.buffer), dtype=np.float32)
    
    def clear(self):
        """Clear the buffer."""
        with self._lock:
            self.buffer.clear()

class VADListener:
    """
    Continuous audio listener with Voice Activity Detection using Hysteresis.
    
    Uses dual threshold approach:
    - start_threshold: High threshold to begin recording (filters noise)
    - end_threshold: Low threshold to maintain recording (captures soft endings)
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 512,
        # Hysteresis thresholds
        start_threshold: float = 0.55,  # High: Hard to open (filters chair noise, clicks)
        end_threshold: float = 0.20,    # Low: Easy to maintain (keeps soft "please")
        min_silence_duration: float = 1.0,  # Time before cutting (reduced because low threshold does the work)
        min_speech_duration: float = 0.3,
        pre_speech_buffer: float = 0.5,
        post_speech_buffer: float = 0.5,
        callback: Optional[Callable[[np.ndarray, int], None]] = None,
    ):
        """
        Initialize VAD listener with hysteresis.
        
        Args:
            sample_rate: Audio sample rate (16kHz recommended for Whisper)
            chunk_size: Size of audio chunks for processing
            start_threshold: High threshold to start recording (0-1)
            end_threshold: Low threshold to maintain recording (0-1)
            min_silence_duration: Minimum silence to end utterance (seconds)
            min_speech_duration: Minimum speech duration to process (seconds)
            pre_speech_buffer: Audio to include before speech starts (seconds)
            post_speech_buffer: Audio to include after speech ends (seconds)
            callback: Optional callback function(audio, sample_rate) for utterances
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.min_silence_duration = min_silence_duration
        self.min_speech_duration = min_speech_duration
        self.post_speech_buffer = post_speech_buffer
        self.callback = callback
        
        # Load Silero VAD model in raw probability mode (no iterator wrapper)
        self.vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True,
        )
        self.vad_model.eval()  # Evaluation mode
        
        # State management
        self.is_running = False
        self.is_speaking = False
        self.last_speech_time = None
        
        # Audio buffers
        self.pre_buffer = AudioBuffer(
            max_duration_seconds=pre_speech_buffer,
            sample_rate=sample_rate,
        )
        self.current_utterance = []
        
        # Queue for processed utterances
        self.utterance_queue: queue.Queue[AudioChunk] = queue.Queue()
        
        logger.info(
            f"VAD Listener initialized: Start>{start_threshold}, Maintain>{end_threshold}"
        )
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """
        Callback for sounddevice input stream.
        
        Uses hysteresis logic: high threshold to start, low threshold to maintain.
        """
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        audio = indata[:, 0].copy().astype(np.float32)
        
        # Get raw probability from Silero (0.0 to 1.0)
        # Silero expects tensor [1, N]
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        
        with torch.no_grad():
            speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()
        
        current_time = time.time()
        
        is_speech_frame = False
        
        if not self.is_speaking:
            if speech_prob >= self.start_threshold:
                is_speech_frame = True
        else:
            if speech_prob >= self.end_threshold:
                is_speech_frame = True
        
        if is_speech_frame:
            if not self.is_speaking:
                logger.debug(f"Speech started (Prob: {speech_prob:.2f})")
                self.is_speaking = True
                
                pre = self.pre_buffer.get_all()
                if len(pre) > 0:
                    self.current_utterance.append(pre)
            
            self.last_speech_time = current_time
            self.current_utterance.append(audio)
            self.pre_buffer.clear()
            
        else:
            # No speech detected on this chunk
            if self.is_speaking:
                self.current_utterance.append(audio)
                
                # Check silence timeout
                silence_duration = current_time - self.last_speech_time
                if silence_duration >= self.min_silence_duration:
                    self._process_utterance(current_time)
            else:
                self.pre_buffer.append(audio)
    
    def _process_utterance(self, timestamp: float):
        """
        Process completed utterance.
        
        Args:
            timestamp: Timestamp when utterance ended
        """
        if not self.current_utterance:
            self.is_speaking = False
            return
        
        utterance_audio = np.concatenate(self.current_utterance)
        duration = len(utterance_audio) / self.sample_rate
        
        if duration < self.min_speech_duration:
            logger.debug(f"Utterance too short: {duration:.2f}s")
            self.current_utterance = []
            self.is_speaking = False
            return
        
        audio_max = np.abs(utterance_audio).max()
        if audio_max < 0.1:
            logger.debug(f"Rejected quiet noise (max={audio_max:.2f})")
            self.current_utterance = []
            self.is_speaking = False
            return
        
        logger.debug(f"Utterance captured: {duration:.2f}s (max={audio_max:.2f})")
        
        chunk = AudioChunk(
            audio=utterance_audio,
            sample_rate=self.sample_rate,
            timestamp=timestamp,
            is_speech=True,
        )
        
        self.utterance_queue.put(chunk)
        
        if self.callback:
            try:
                self.callback(utterance_audio, self.sample_rate)
            except Exception as e:
                logger.error(f"Error in callback: {e}", exc_info=True)
        
        self.current_utterance = []
        self.is_speaking = False
    
    def start(self):
        if self.is_running:
            logger.warning("VAD Listener already running")
            return
        
        self.is_running = True
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            callback=self._audio_callback,
            dtype=np.float32,
        )
        
        self.stream.start()
    
    def stop(self):
        if not self.is_running:
            return
        
        self.is_running = False
        
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        # Process any remaining utterance
        if self.current_utterance:
            self._process_utterance(time.time())
        
    def get_utterance(self, timeout: Optional[float] = None) -> Optional[AudioChunk]:
        """
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            AudioChunk or None if timeout
        """
        try:
            return self.utterance_queue.get(timeout=timeout)
        except queue.Empty:
            return None
