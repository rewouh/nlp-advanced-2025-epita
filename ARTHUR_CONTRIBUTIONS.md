# Arthur HAMARD - Project Contributions 8)

This document describes the work I contributed to the **Overhearing Agents** project.
My primary contribution was building the **NPC Conversational Orchestrator**.

For that, I designed and implemented the dialogue system that allows NPCs to have meaningful conversations with players.

The orchestrator takes player speech (already transcribed by our STT module) and generates contextual, personality-driven NPC responses. NPCs actually remember conversations, track their relationship with the player, and adjust their behavior accordingly.

### Key features

**Pydantic Data Contracts**

Definition of clean input/output schemas to make integration with the rest of the pipeline straightforward:

- `NPCInput` - Everything the orchestrator needs: NPC identity, persona, world state, player's words, and any relevant lore from RAG
- `NPCOutput` - The response text (ready for TTS), conversation history, relationship score, and whether the response was interrupted

**Persona System**

Each NPC can have:
- Personality traits (mysterious, grumpy, friendly...)
- Motives that drive their behavior
- Private knowledge they might share - but only if they trust the player

**Relationship Tracker**

The system detects sentiment in player speech:

- Rude patterns ("shut up", "idiot", etc.) damage the relationship (-15 points)
- Polite patterns ("please", "thank you") improve it (+10 points)

The score ranges from -100 (hostile) to +100 (trusting) and affects how NPCs behave:
- Hostile NPCs become dismissive or threatening
- Trusting NPCs share their secrets freely

**Memory Management with Summarization**

Use of LangChain's `ConversationSummaryBufferMemory` to handle long conversations. Instead of feeding the entire chat history to the LLM (which would blow up context), it keeps recent exchanges verbatim while summarizing older ones. This lets conversations feel continuous without hitting token limits.

**Streaming Generation with Interruption**

The `think()` method yields tokens as they're generated, enabling concurrent TTS playback. Players can interrupt mid-response (useful when the NPC is rambling), and the system handles it gracefully with proper cleanup.

**Thread-Safe Design**

Since we're dealing with real-time audio, the orchestrator is thread-safe. Relationship updates and generation state use locks, and there's a proper interrupt flag mechanism.

### System Prompt Engineering

Detailed prompt template that injects:
- NPC identity and personality
- Current scene context (location, time, mood)
- Retrieved lore from the RAG module
- Relationship status and corresponding behavioral instructions
- Conversation summary

The prompt tells the NPC to stay in character, be autonomous (can refuse to help, be evasive, etc.), and respond naturally without the typical asterisk actions.

### Test Suite

I also wrote a comprehensive test file with:
- Unit tests for the relationship tracker and Pydantic schemas
- Integration tests that actually spin up the LLM
- A demo tavern scenario with a "dark-shaped figure" NPC
- Tests for interruption handling and social dynamics
