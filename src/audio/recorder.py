"""
Simple audio recorder for testing and development.
"""

import logging
import tempfile
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import pyaudio

logger = logging.getLogger(__name__)


class AudioRecorder:
    """Simple audio recorder for testing."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
    ):
        """
        Initialize audio recorder.
        
        Args:
            sample_rate: Sample rate in Hz
            channels: Number of channels (1=mono, 2=stereo)
            chunk_size: Chunk size for recording
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = pyaudio.paInt16
    
    def record(
        self,
        duration: float,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Record audio for specified duration.
        
        Args:
            duration: Recording duration in seconds
            output_path: Optional output path (creates temp file if None)
            
        Returns:
            Path to recorded WAV file
        """
        p = pyaudio.PyAudio()
        
        stream = p.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )
        
        logger.info(f"Recording for {duration}s...")
        
        frames = []
        num_chunks = int(self.sample_rate / self.chunk_size * duration)
        
        for _ in range(num_chunks):
            data = stream.read(self.chunk_size)
            frames.append(data)
        
        logger.info("Recording complete")
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save to file
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav",
            )
            output_path = Path(temp_file.name)
        
        with wave.open(str(output_path), 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(p.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
        
        return output_path
    
    @staticmethod
    def numpy_to_wav(
        audio: np.ndarray,
        sample_rate: int,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Save numpy array as WAV file.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate
            output_path: Optional output path
            
        Returns:
            Path to saved WAV file
        """
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav",
            )
            output_path = Path(temp_file.name)
        
        # Convert float32 to int16
        if audio.dtype == np.float32:
            audio = (audio * 32767).astype(np.int16)
        
        with wave.open(str(output_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
        
        return output_path

