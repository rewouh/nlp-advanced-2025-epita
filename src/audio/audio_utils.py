import numpy as np
import tempfile
import wave

from pathlib import Path
from typing import Optional

class AudioUtils:
        
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
        
        if audio.dtype == np.float32:
            audio = (audio * 32767).astype(np.int16)
        
        with wave.open(str(output_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
        
        return output_path
