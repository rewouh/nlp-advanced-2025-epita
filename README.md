# Overhearing Agents

Real-time NPC conversation system for tabletop RPGs using local LLMs, RAG, and voice detection.

## Quick Start

```bash
# Install dependencies
uv sync  # or pip install -e .

# Install Ollama model
ollama pull qwen2.5:3b

# Run full system (with audio)
COQUI_TOS_AGREED=1 uv run python -m src.main
```

## Architecture

The system implements a multi-threaded real-time pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│                    Overhearing Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [Thread 1] VAD Listener                                     │
│    └─> Continuous microphone capture                         │
│    └─> Voice Activity Detection (hysteresis)                │
│    └─> Audio chunk queue                                     │
│                                                              │
│  [Thread 2] STT Worker                                       │
│    └─> faster-whisper transcription                         │
│    └─> Hotword support (NPC names, locations, RPG terms)     │
│    └─> Transcription queue                                  │
│                                                              │
│  [Thread 3] Overhearing Worker                               │
│    └─> Context Manager (location, time, mood)               │
│    └─> Trigger Detector (sticky context, direct/indirect)  │
│    └─> NPC activation                                       │
│                                                              │
│  [Main Thread] NPC Pipeline                                 │
│    └─> RAG retrieval (ChromaDB + world lore)                │
│    └─> NPC Orchestrator (LangChain + Ollama)                │
│    └─> TTS Engine (Coqui XTTS v2)                           │
│        └─> Emotion support                                   │
│        └─> Voice cloning (unique per NPC)                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

- **VAD Listener**: Dual-threshold hysteresis for natural speech detection
- **STT Engine**: GPU-accelerated transcription with custom vocabulary
- **Context Manager**: Dynamic state tracking (location, time, mood, NPCs)
- **Trigger Detector**: Active window (sticky context) for natural conversation flow
- **NPC Pipeline**: RAG-enhanced NPC responses with personality and memory
- **TTS Engine**: Emotion-aware speech synthesis with voice cloning

## Tests

```bash
# Unit tests
python tests/test_stt.py
python tests/test_orchestrator.py
python tests/test_rag.py

# End-to-end
python tests/test_e2e_pipeline.py
```

For testing the full system with example conversations and questions, see [TESTING_GUIDE.md](TESTING_GUIDE.md).

## Project Structure

- `src/stt/` - Speech-to-text with hotwords
- `src/audio/` - VAD and audio processing
- `src/overhearing/` - Context tracking and trigger detection
- `src/pipeline/` - Integration of all components
- `src/conversation/` - NPC orchestrator (LangChain)
- `src/rag/` - Knowledge retrieval (ChromaDB)
- `lore/` - World and session data

## Authors

Léo S, Arthur G, Arthur H, Pierre B, Etienne S
