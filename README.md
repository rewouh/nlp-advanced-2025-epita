# Overhearing Agents

Real-time NPC conversation system for tabletop RPGs using local LLMs, RAG, and voice detection.

#### What's a tabletop RPG ?

It's a game where players create characters and collaboratively tell a story, guided by rules and a game master (who created the universe and continously builds the story), usually using dice to determine outcomes.

In regular tabletop RPGs, the game master is generally the one that embodies the NPCs (Non-Player-Characters, persons the players can encounter during their travels, ex : a barkeeper) by changing his tone and his way of talking.

Yet, we know that LLMs are very good at speaking in a certain manner, they can easily embody peoples, creatures, anything and give very creative outputs.

That is why we decided to create a system where NPCs are personified by locally-powered LLMs, with whom players can discuss.

## Members

`Léo Sambrook`, `Arthur Guelennoc`, `Arthur Hamard`, `Pierre Braud`, `Etienne Senigout`

## Quick Start

```bash
# Dependencies
uv sync 

# Ollama model
ollama pull qwen2.5:3b

# Run
uv run python -m src.main
```

## Architecture

The system implements a multi-threaded real-time pipeline:

```
┌──────────────────────────────────────────────────────────────┐
│                    Overhearing Pipeline                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  [Thread 1] VAD Listener                                     │
│    └─> Continuous microphone capture                         │
│    └─> Voice Activity Detection (hysteresis)                 │
│    └─> Audio chunk queue                                     │
│                                                              │
│  [Thread 2] STT Worker                                       │
│    └─> faster-whisper transcription                          │
│    └─> Hotword support (NPC names, locations, RPG terms)     │
│    └─> Transcription queue                                   │
│                                                              │
│  [Thread 3] Overhearing Worker                               │
│    └─> Context Manager (location, time, mood)                │
│    └─> Trigger Detector (sticky context, direct/indirect)    │
│    └─> NPC activation                                        │
│                                                              │
│  [Main Thread] NPC Pipeline                                  │
│    └─> RAG retrieval (ChromaDB + world lore)                 │
│    └─> NPC Orchestrator (LangChain + Ollama)                 │
│    └─> TTS Engine (Coqui XTTS v2)                            │
│        └─> Emotion support                                   │
│        └─> Voice cloning (unique per NPC)                    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Key Components

- **VAD Listener**: Dual-threshold hysteresis for natural speech detection
- **STT Engine**: GPU-accelerated transcription with custom vocabulary
- **Context Manager**: Dynamic state tracking (location, time, mood, NPCs)
- **Trigger Detector**: Active window (sticky context) for natural conversation flow
- **NPC Pipeline**: RAG-enhanced NPC responses with personality and memory
- **TTS Engine**: Emotion-aware speech synthesis with voice cloning

## Tests

The command below runs the testsuite (can take a bit of time!).
```bash
uv run python -m src.main --eval
```

## Project Structure

- `src/stt/` - Speech-to-text with hotwords
- `src/audio/` - VAD and audio processing
- `src/overhearing/` - Context tracking and trigger detection
- `src/pipeline/` - Integration of all components
- `src/conversation/` - NPC orchestrator (LangChain)
- `src/rag/` - Knowledge retrieval (ChromaDB)
- `lore/` - World and session data
