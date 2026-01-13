"""
NPC Conversational Orchestrator Module

Manages active dialogue between players and NPCs using LangChain and Ollama.
Handles persona integration, memory orchestration with summarization,
interruption handling, and social dynamics tracking.
"""

from typing import List, Generator
from pydantic import BaseModel, Field
import threading
import time

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_classic.memory.summary_buffer import ConversationSummaryBufferMemory


class Persona(BaseModel):
    """NPC personality and background information."""

    traits: List[str] = Field(
        default_factory=list,
        description="Personality traits (e.g., shy, playful, grumpy)",
    )
    motives: str = Field(
        default="", description="What drives this NPC's actions and decisions"
    )
    private_knowledge: str = Field(
        default="", description="Information the NPC knows but may not share freely"
    )


class GlobalState(BaseModel):
    """Current world/scene state affecting NPC behavior."""

    location: str = Field(
        default="unknown", description="Current location (e.g., tavern, forest, castle)"
    )
    time_of_day: str = Field(
        default="day", description="Time of day (morning, day, evening, night)"
    )
    scene_mood: str = Field(
        default="neutral",
        description="Overall mood of the scene (tense, relaxed, mysterious)",
    )


class NPCInput(BaseModel):
    """Input data structure for NPC interaction."""

    npc_id: str = Field(
        ..., description="Unique identifier for the NPC (e.g., 'pirate_captain')"
    )
    persona: Persona = Field(
        default_factory=Persona,
        description="NPC's personality, motives, and private knowledge",
    )
    global_state: GlobalState = Field(
        default_factory=GlobalState,
        description="Current world state affecting NPC behavior",
    )
    player_input: str = Field(
        ..., description="Transcribed text from STT (what the player said)"
    )
    retrieved_lore: str = Field(
        default="", description="Mocked output from future RAG module"
    )


class NPCOutput(BaseModel):
    """Output data structure from NPC interaction."""

    agent_response: str = Field(..., description="NPC's response text for TTS")
    updated_history: List[dict] = Field(
        default_factory=list, description="Serialized conversation state"
    )
    relationship_score: int = Field(
        default=0, description="Current relationship score (-100 to 100)"
    )
    was_interrupted: bool = Field(
        default=False, description="Whether the response was interrupted"
    )


class RelationshipTracker:
    """
    Tracks player-NPC relationship based on interaction patterns.
    Influences NPC's willingness to share information.
    Score ranges from -100 (hostile) to +100 (trusting).
    """

    RUDE_PATTERNS = [
        "shut up",
        "idiot",
        "stupid",
        "fool",
        "hate",
        "ugly",
        "worthless",
        "useless",
        "die",
        "kill",
        "damn",
        "hell",
        "get lost",
        "go away",
        "leave me",
        "annoying",
    ]

    HELPFUL_PATTERNS = [
        "please",
        "thank",
        "thanks",
        "appreciate",
        "kind",
        "help",
        "sorry",
        "excuse",
        "friend",
        "lovely",
        "wonderful",
        "great",
        "amazing",
        "beautiful",
    ]

    def __init__(self, initial_score: int = 0):
        self.score = initial_score
        self._lock = threading.Lock()

    def analyze_input(self, text: str) -> int:
        """Analyze player input and update relationship score. Returns delta applied."""
        text_lower = text.lower()
        delta = 0

        for pattern in self.RUDE_PATTERNS:
            if pattern in text_lower:
                delta -= 15
                break

        for pattern in self.HELPFUL_PATTERNS:
            if pattern in text_lower:
                delta += 10
                break

        with self._lock:
            self.score = max(-100, min(100, self.score + delta))

        return delta

    def get_score(self) -> int:
        with self._lock:
            return self.score

    def get_disposition(self) -> str:
        """Get NPC's current disposition based on relationship score."""
        score = self.get_score()
        if score <= -50:
            return "hostile"
        elif score <= -20:
            return "unfriendly"
        elif score <= 20:
            return "neutral"
        elif score <= 50:
            return "friendly"
        else:
            return "trusting"

    def should_share_secrets(self) -> bool:
        """Determine if NPC is willing to share private knowledge."""
        return self.get_score() >= 40


class NPCOrchestrator:
    """
    Main orchestrator for NPC conversations.
    Handles persona integration, memory management, and response generation.
    """

    SYSTEM_TEMPLATE = """You are {npc_id}, a character in a fantasy roleplay world.

## Your Persona
Traits: {traits}
Motives: {motives}
{private_knowledge_section}

## Current Situation
Location: {location}
Time: {time_of_day}
Mood: {scene_mood}

## Relevant Lore
{retrieved_lore}

## Your Relationship with this Player
Current disposition: {disposition}
{relationship_instructions}

## Instructions
- Stay completely in character at all times.
- You are autonomous: you can choose to withhold information, be evasive, or interrupt the player if it fits your persona.
- Keep responses natural and concise (1-3 sentences typically).
- No actions in *asterisks* - only dialogue.
- React based on your personality traits and current disposition toward the player.
- If the player is being rude, you may become less cooperative or even refuse to help.
- If asked about your private knowledge, only share if you trust this player.
- If your private knowledge mentions items you possess, you can physically give those items to the player if you trust them and they ask for help. Clearly state that you are giving them the item (e.g., "Here, take my [item name]" or "I'm giving you [item name]").

Previous conversation summary:
{summary}"""

    def __init__(
        self,
        npc_id: str,
        persona: Persona,
        model: str = "qwen2.5:3b",
        temperature: float = 0.7,
        max_token_limit: int = 500,
        num_recent_exchanges: int = 5,
    ):
        self.npc_id = npc_id
        self.persona = persona
        self.model_name = model
        self.num_recent_exchanges = num_recent_exchanges

        print(f"Loading NPC Orchestrator with model: {model}...")
        self.llm = ChatOllama(model=model, temperature=temperature, num_predict=256)

        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=max_token_limit,
            memory_key="history",
            return_messages=True,
            human_prefix="Player",
            ai_prefix=npc_id,
        )

        self.relationship = RelationshipTracker()

        self._interrupt_flag = threading.Event()
        self._generation_lock = threading.Lock()
        self._is_generating = False

        self.global_state = GlobalState()
        self.retrieved_lore = ""

    def update_state(self, global_state: GlobalState):
        """Update the current global state."""
        self.global_state = global_state

    def set_lore(self, lore: str):
        """Set retrieved lore from RAG module."""
        self.retrieved_lore = lore

    def interrupt(self):
        """Signal to interrupt ongoing generation."""
        self._interrupt_flag.set()

    def _build_system_prompt(self) -> str:
        traits_str = (
            ", ".join(self.persona.traits) if self.persona.traits else "none specified"
        )

        if self.relationship.should_share_secrets() and self.persona.private_knowledge:
            private_knowledge_section = (
                f"Private Knowledge (you may share): {self.persona.private_knowledge}"
            )
        elif self.persona.private_knowledge:
            private_knowledge_section = f"Private Knowledge (keep secret unless you really trust them): {self.persona.private_knowledge}"
        else:
            private_knowledge_section = ""

        disposition = self.relationship.get_disposition()
        relationship_instructions = {
            "hostile": "You are hostile toward this player. Be dismissive, unhelpful, or even threatening.",
            "unfriendly": "You are wary of this player. Be curt and reluctant to help.",
            "neutral": "You have no strong feelings about this player. Be professional but not overly warm.",
            "friendly": "You like this player. Be warm, helpful, and open.",
            "trusting": "You trust this player deeply. Be open, share freely, and show genuine warmth.",
        }.get(disposition, "")

        memory_vars = self.memory.load_memory_variables({})
        summary = memory_vars.get("history", "No previous conversation.")
        if isinstance(summary, list):
            summary_parts = []
            for msg in summary[-self.num_recent_exchanges * 2 :]:
                if isinstance(msg, HumanMessage):
                    summary_parts.append(f"Player: {msg.content}")
                elif isinstance(msg, AIMessage):
                    summary_parts.append(f"{self.npc_id}: {msg.content}")
            summary = (
                "\n".join(summary_parts)
                if summary_parts
                else "No previous conversation."
            )

        return self.SYSTEM_TEMPLATE.format(
            npc_id=self.npc_id,
            traits=traits_str,
            motives=self.persona.motives or "unknown",
            private_knowledge_section=private_knowledge_section,
            location=self.global_state.location,
            time_of_day=self.global_state.time_of_day,
            scene_mood=self.global_state.scene_mood,
            retrieved_lore=self.retrieved_lore or "None available.",
            disposition=disposition,
            relationship_instructions=relationship_instructions,
            summary=summary,
        )

    def think(self, player_input: str) -> Generator[str, None, None]:
        """
        Generate NPC response with streaming support.
        Yields tokens as they're generated. Can be interrupted via interrupt().
        """
        self._interrupt_flag.clear()

        with self._generation_lock:
            self._is_generating = True

        try:
            self.relationship.analyze_input(player_input)

            system_prompt = self._build_system_prompt()

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=player_input),
            ]

            full_response = ""
            start_time = time.time()

            for chunk in self.llm.stream(messages):
                if self._interrupt_flag.is_set():
                    print(f"\n[Interrupted after {time.time() - start_time:.2f}s]")
                    break

                content = chunk.content
                token = content if isinstance(content, str) else str(content)
                full_response += token
                yield token

            if full_response:
                self.memory.save_context(
                    {"input": player_input}, {"output": full_response}
                )

            elapsed = time.time() - start_time
            print(f"\n[Generation completed in {elapsed:.2f}s]")

        finally:
            with self._generation_lock:
                self._is_generating = False

    def think_sync(self, player_input: str) -> str:
        """Synchronous version of think() that returns complete response."""
        tokens = list(self.think(player_input))
        return "".join(tokens)

    def process_input(self, npc_input: NPCInput) -> NPCOutput:
        """
        Process a complete NPCInput and return NPCOutput.
        This is the main interface matching the data contract.
        """
        self.persona = npc_input.persona
        self.global_state = npc_input.global_state
        self.retrieved_lore = npc_input.retrieved_lore

        was_interrupted = False
        response = self.think_sync(npc_input.player_input)

        if self._interrupt_flag.is_set():
            was_interrupted = True

        memory_vars = self.memory.load_memory_variables({})
        history = memory_vars.get("history", [])

        serialized_history = []
        if isinstance(history, list):
            for msg in history:
                if isinstance(msg, HumanMessage):
                    serialized_history.append(
                        {"role": "player", "content": msg.content}
                    )
                elif isinstance(msg, AIMessage):
                    serialized_history.append({"role": "npc", "content": msg.content})

        return NPCOutput(
            agent_response=response,
            updated_history=serialized_history,
            relationship_score=self.relationship.get_score(),
            was_interrupted=was_interrupted,
        )

    def get_history_serialized(self) -> List[dict]:
        """Get serialized conversation history."""
        memory_vars = self.memory.load_memory_variables({})
        history = memory_vars.get("history", [])

        serialized = []
        if isinstance(history, list):
            for msg in history:
                if isinstance(msg, HumanMessage):
                    serialized.append({"role": "player", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    serialized.append({"role": "npc", "content": msg.content})

        return serialized

    def reset_memory(self):
        """Clear conversation memory."""
        self.memory.clear()

    def reset_relationship(self):
        """Reset relationship score to neutral."""
        self.relationship = RelationshipTracker()


def create_npc(
    npc_id: str,
    traits: List[str],
    motives: str = "",
    private_knowledge: str = "",
    model: str = "qwen2.5:3b",
) -> NPCOrchestrator:
    """Factory to create an NPC orchestrator with simplified parameters."""
    persona = Persona(
        traits=traits, motives=motives, private_knowledge=private_knowledge
    )
    return NPCOrchestrator(npc_id=npc_id, persona=persona, model=model)
