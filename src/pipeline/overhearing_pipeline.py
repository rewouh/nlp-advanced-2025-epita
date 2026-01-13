import logging
import queue
import threading
import time
import requests
import numpy as np

from pathlib import Path
from typing import Optional, Callable
from src.audio import VADListener, AudioUtils
from src.stt import STTEngine, create_stt_engine
from src.overhearing import ContextManager, TriggerDetector, TriggerType
from src.pipeline.npc_pipeline import NPCPipeline
from src.rag.lore_manager import LoreManager

logger = logging.getLogger(__name__)

class OverhearingPipeline:
    """
    Complete overhearing pipeline with multi-threading.
    
    Architecture:
    [Thread 1] VAD Listener → Audio Queue
    [Thread 2] STT Processor → Transcription Queue
    [Thread 3] Overhearing Logic → NPC Activation
    [Main] NPC Response + TTS
    """
    
    def __init__(
        self,
        world_path: Path,
        session_path: Optional[Path] = None,
        model: str = "qwen2.5:3b",
        stt_model: str = "base",
        tts_callback: Optional[Callable[[str, Optional[str], Optional[str]], None]] = None,
    ):
        """
        Initialize overhearing pipeline.
        
        Args:
            world_path: Path to world.yaml
            session_path: Optional path to session.yaml
            model: LLM model for NPC orchestrators
            stt_model: Whisper model size (tiny, base, small, medium)
            tts_callback: Optional callback for TTS (receives text)
        """
        self.lore_manager = LoreManager(persist_directory="./data/chroma")
        self.world = self.lore_manager.load_world(world_path)
        
        if session_path and session_path.exists():
            self.session = self.lore_manager.load_session(session_path)
        else:
            from src.rag.models import SessionState
            self.session = SessionState()
        
        self.stt_engine = create_stt_engine(
            world_lore=self.world,
            model_size=stt_model,
            device="auto",
            compute_type="auto",
        )
        
        self.context_manager = ContextManager(
            initial_session=self.session,
            world_lore=self.world
        )
        
        self.trigger_detector = TriggerDetector(world_lore=self.world)
        
        self.npc_pipeline = NPCPipeline(
            lore_manager=self.lore_manager,
            model=model,
        )
        
        self.vad_listener = VADListener(
            sample_rate=16000,
            # Hysteresis settings
            start_threshold=0.55,  # High threshold to filter ambient noise
            end_threshold=0.20,    # Low threshold to maintain recording
            min_silence_duration=1.0,  # Minimum silence before ending utterance
            min_speech_duration=0.3,
            pre_speech_buffer=0.5,
            post_speech_buffer=0.5,
        )
        
        self.tts_callback = tts_callback
        
        self.transcription_queue: queue.Queue[str] = queue.Queue()
        
        # Control flags
        self.is_running = False
        self._threads = []
        
        self._is_speaking_lock = threading.Lock()
        self._is_speaking = False
        
        self._last_npc_speech_end_time = 0.0
        self._last_active_npc: Optional[str] = None
    
    def start(self):
        """Start the pipeline."""
        if self.is_running:
            logger.warning("Pipeline already running")
            return
        
        self.is_running = True
        self.vad_listener.start()
        self._threads = [
            threading.Thread(target=self._stt_worker, daemon=True),
            threading.Thread(target=self._overhearing_worker, daemon=True),
        ]
        
        for thread in self._threads:
            thread.start()
    
    def stop(self):
        """Stop the pipeline."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.vad_listener.stop()
        
        for thread in self._threads:
            thread.join(timeout=2.0)
    
    def _stt_worker(self):
        """
        STT worker thread.
        
        Continuously processes audio from VAD listener and transcribes.
        """
        while self.is_running:
            audio_chunk = self.vad_listener.get_utterance(timeout=1.0)
            if audio_chunk is None:
                continue
            
            with self._is_speaking_lock:
                if self._is_speaking:
                    logger.debug("Skipping: NPC is speaking")
                    continue
                
                time_since_speech = time.time() - self._last_npc_speech_end_time
                if time_since_speech < 1.5:
                    logger.debug(f"Skipping: Audio Echo Cooldown ({time_since_speech:.2f}s < 1.5s)")
                    continue
            
            try:
                duration = len(audio_chunk.audio) / audio_chunk.sample_rate
                
                if duration < 0.3:
                    logger.debug(f"Skipping short audio chunk ({duration:.2f}s)")
                    continue
                
                logger.debug(f"Processing audio chunk ({duration:.2f}s)")
                
                audio_max = np.abs(audio_chunk.audio).max()
                if audio_max < 0.4:
                    logger.debug(f"Skipping quiet audio chunk (max={audio_max:.4f})")
                    continue
                
                temp_path = AudioUtils.numpy_to_wav(audio_chunk.audio, audio_chunk.sample_rate)
                transcription = self.stt_engine.transcribe(
                    temp_path,
                    language="en",
                    vad_filter=False,
                )
                
                temp_path.unlink()
                
                if transcription and transcription.strip():
                    logger.info(f"Transcribed: \"{transcription}\"")
                    self.transcription_queue.put(transcription)
                else:
                    logger.warning(f"Empty transcription for {duration:.2f}s audio chunk (max amplitude: {audio_max:.4f})")
                
            except Exception as e:
                logger.error(f"STT error: {e}", exc_info=True)
    
    def _overhearing_worker(self):
        """
        Overhearing logic worker thread.
        
        Processes transcriptions, detects triggers, and activates NPCs.
        """
        while self.is_running:
            try:
                transcription = self.transcription_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            try:
                self.context_manager.add_conversation_turn(
                    speaker="player",
                    text=transcription,
                )
                
                context_changes = self.context_manager.detect_context_changes(
                    transcription
                )
                
                if context_changes:
                    logger.debug(f"Context changes detected: {context_changes}")
                    
                    if "location" in context_changes:
                        new_location_id = context_changes["location"]
                        
                        self.trigger_detector.reset_state()
                        self._last_active_npc = None
                        
                        npcs_added = []
                        for npc in self.world.npcs:
                            if npc.location == new_location_id:
                                self.context_manager.add_npc_to_scene(npc.id)
                                npcs_added.append(npc.id)
                        
                    
                    self.context_manager.apply_detected_changes(context_changes)
                
                context = self.context_manager.get_context_snapshot()
                
                # If we just had an NPC response, boost direct trigger confidence
                # (player is likely responding to the NPC)
                boost_direct = False
                if self._last_active_npc and self._last_active_npc in context.active_npcs:
                    # Check if this looks like a response (short, no clear trigger)
                    if len(transcription.split()) < 10:
                        boost_direct = True
                
                recent_history = self.context_manager.get_recent_history(n=5)
                
                if recent_history:
                    logger.debug(f"Recent history: {len(recent_history)} turns")
                    for i, turn in enumerate(recent_history[-3:]):  # Last 3 turns
                        logger.debug(f"  Turn {i}: {turn.speaker} ({getattr(turn, 'npc_id', 'N/A')}): {turn.text[:50]}")
                
                trigger_result = self.trigger_detector.detect_trigger(
                    transcription,
                    active_npcs=context.active_npcs,
                    boost_direct=boost_direct,
                    conversation_history=recent_history,
                )
                
                logger.info(
                    f"Trigger: {trigger_result.trigger_type.value} "
                    f"(confidence: {trigger_result.confidence:.2f}) - {trigger_result.reasoning}"
                )
                
                # Handle trigger
                if trigger_result.trigger_type in [
                    TriggerType.NPC_DIRECT,
                    TriggerType.NPC_INDIRECT,
                ]:
                    if trigger_result.triggered_npc:
                        if trigger_result.triggered_npc in context.active_npcs:
                            self._last_active_npc = trigger_result.triggered_npc
                            self._handle_npc_activation(
                                npc_id=trigger_result.triggered_npc,
                                player_input=transcription,
                                context=context,
                            )
                        
                        elif trigger_result.trigger_type == TriggerType.NPC_DIRECT:
                            logger.info(
                                f"Summoning NPC {trigger_result.triggered_npc} to scene (Direct Trigger)"
                            )
                            
                            self.context_manager.add_npc_to_scene(trigger_result.triggered_npc)
                            
                            context = self.context_manager.get_context_snapshot()
                            
                            self._last_active_npc = trigger_result.triggered_npc
                            self._handle_npc_activation(
                                npc_id=trigger_result.triggered_npc,
                                player_input=transcription,
                                context=context,
                            )
                        
                        else:
                            logger.warning(
                                f"NPC {trigger_result.triggered_npc} triggered (indirect) but not in scene. "
                                f"Active NPCs: {context.active_npcs}"
                            )
                
            except Exception as e:
                logger.error(f"Overhearing worker error: {e}", exc_info=True)
    
    def _handle_npc_activation(self, npc_id: str, player_input: str, context):
        """
        Handle NPC activation (run in main thread or separate thread).
        
        Args:
            npc_id: NPC to activate
            player_input: Player's input
            context: Current game context
        """
        logger.info(f"=== NPC ACTIVATION: {npc_id} ===")
        
        try:
            self.context_manager.add_npc_to_scene(npc_id)
            
            response = self.npc_pipeline.activate_npc(
                npc_id=npc_id,
                player_input=player_input,
                current_location=context.current_location,
                time_of_day=context.time_of_day,
                scene_mood=context.scene_mood,
            )

            requests.post("http://localhost:5000/say",
                json={"who": npc_id, "text": response}
            )
            
            self.context_manager.add_conversation_turn(
                speaker="npc",
                text=response,
                npc_id=npc_id,
            )
            
            if self.tts_callback:
                orchestrator = self.npc_pipeline.active_npcs.get(npc_id)
                disposition = orchestrator.relationship.get_disposition() if orchestrator else "neutral"
                
                npc_def = self.lore_manager.get_npc_definition(npc_id)
                voice_gender = None
                if npc_def and hasattr(npc_def, 'voice_gender'):
                    voice_gender = npc_def.voice_gender
                
                if not voice_gender:
                    voice_gender = self._detect_gender_from_name(npc_id, npc_def)
                
                logger.debug(f"{npc_id} responding ({disposition}, voice: {voice_gender or 'auto'})...")
                
                with self._is_speaking_lock:
                    self._is_speaking = True
                
                try:
                    # Pass emotion, npc_id, and voice_gender to callback
                    try:
                        self.tts_callback(response, emotion=disposition, npc_id=npc_id, voice_gender=voice_gender)
                    except TypeError:
                        try:
                            self.tts_callback(response, emotion=disposition, npc_id=npc_id)
                        except TypeError:
                            try:
                                self.tts_callback(response, emotion=disposition)
                            except TypeError:
                                self.tts_callback(response)
                    
                    self.trigger_detector.update_state_after_speech(npc_id)
                finally:
                    with self._is_speaking_lock:
                        self._is_speaking = False
                        self._last_npc_speech_end_time = time.time()
                    logger.debug("Listening re-enabled after NPC finished speaking")
            
        except Exception as e:
            logger.error(f"NPC activation error: {e}", exc_info=True)
    
    def process_input_manually(self, text: str):
        """
        Args:
            text: Text to process
        """
        self.transcription_queue.put(text)
    
    def get_context_summary(self) -> str:
        """
        Returns:
            Human-readable context summary
        """
        return self.context_manager.get_context_summary()
    
    def set_location(self, location: str, location_id: Optional[str] = None):
        """
        Args:
            location: Location name
            location_id: Optional location ID
        """
        self.context_manager.update_location(location, location_id)
    
    def add_npc_to_scene(self, npc_id: str):
        """
        Args:
            npc_id: NPC identifier
        """
        self.context_manager.add_npc_to_scene(npc_id)
    
    def _detect_gender_from_name(self, npc_id: str, npc_def) -> Optional[str]:
        """
        Args:
            npc_id: NPC identifier
            npc_def: NPC definition (optional)
            
        Returns:
            'male' or 'female'
        """
        # Common female name patterns
        female_patterns = [
            'hilda', 'hannah', 'sarah', 'mary', 'anna', 'emma', 'sophia',
            'elizabeth', 'catherine', 'diana', 'helen', 'jane', 'lisa',
            'patricia', 'nancy', 'susan', 'karen', 'betty', 'dorothy'
        ]
        
        # Common female indicators in backstory
        female_indicators = ['she', 'her', 'woman', 'lady', 'female']
        
        name = ""
        backstory = ""
        if npc_def:
            name = (npc_def.name or "").lower()
            backstory = (npc_def.backstory or "").lower()
        
        name_lower = name or npc_id.lower()
        for pattern in female_patterns:
            if pattern in name_lower:
                return "female"
        
        if backstory:
            for indicator in female_indicators:
                if indicator in backstory:
                    return "female"
        
        return "male"
