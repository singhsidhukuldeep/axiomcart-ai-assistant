"""
AxiomCart Multi-Agent Voice System — main entry point.

Usage:
    uv run python -m src.main                # interactive text mode (default)
    uv run python -m src.main --voice        # interactive voice mode (needs mic)
    uv run python -m src.main --query "..."  # single text query
"""

from __future__ import annotations # for newer type hint syntax on older Python version (<3.10+)

import argparse
import uuid

from langgraph.types import Command

from langchain_core.messages import HumanMessage

from src.config import get_logger
from src.graph import axiomcart_graph
from src.voice import VoiceRecorder, VoiceSpeaker

logger = get_logger("main")


class AxiomCartAssistant:
    """Wraps the LangGraph multi-agent pipeline with optional voice I/O.

    A single thread_id is used for the entire session so the
    checkpointer carries conversation history across turns.
    """

    def __init__(self, voice: str = "nova", enable_voice: bool = False):
        self.enable_voice = enable_voice
        self.thread_id = uuid.uuid4().hex  # single thread per session

        if enable_voice:
            self.recorder = VoiceRecorder()
            self.speaker  = VoiceSpeaker(voice=voice, speed=1.1)
        else:
            self.recorder = None
            self.speaker  = None

        logger.info("Assistant ready  (voice=%s)", enable_voice)

    # ── core query ────────────────────────────────────────
    # ── core query (handles HITL interrupt/resume) ───────
    def query(self, text: str, input_fn=None) -> str:
        """Send a text query through the multi-agent graph.

        If the graph pauses for HITL (support agent needs order ID),
        this method collects user input and resumes the graph.
        """
        if input_fn is None:
            input_fn = lambda prompt: input(f"\n🔄 Agent asks: {prompt}\nYou: ").strip()

        config = {"configurable": {"thread_id": self.thread_id}}

        result = axiomcart_graph.invoke(
            {"messages": [HumanMessage(content=text)], "user_query": text},
            config,
        )

        # Handle HITL (Human-in-the-Loop)
        # interrupt() surfaces questions in __interrupt__
        while "__interrupt__" in result and result["__interrupt__"]:
            question = result["__interrupt__"][0].value
            logger.info("HITL interrupt: %r", question)

            if self.enable_voice and self.speaker and self.recorder:
                self.speaker.speak(question)
                _, user_answer = self.recorder.record_and_transcribe(duration=5)
                if not user_answer:
                    user_answer = "I don't have that information"
            else:
                user_answer = input_fn(question)

            logger.info("HITL resume: user_answer=%r", user_answer)
            result = axiomcart_graph.invoke(Command(resume=user_answer), config)

        answer = result.get("final_answer", "")
        if not answer:
            answer = "Sorry, I wasn't able to process that. Could you try rephrasing?"
        return answer

    # ── interactive loops ────────────────────────────────
    def text_loop(self) -> None:
        """REPL-style text interaction."""
        print("\n🛒  AxiomCart Assistant  (type 'quit' to exit)\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input or user_input.lower() in ("quit", "exit", "bye"):
                print("Goodbye!")
                break
            answer = self.query(user_input)
            print(f"\nAssistant: {answer}\n")

    def voice_loop(self, max_turns: int = 10) -> None:
        """Microphone-based conversation loop."""
        if not self.recorder or not self.speaker:
            logger.error("Voice components not initialised")
            return

        welcome = "Hello! I'm your AxiomCart assistant. How can I help you today?"
        self.speaker.speak(welcome)

        for turn in range(1, max_turns + 1):
            logger.info("--- voice turn %d / %d ---", turn, max_turns)
            _, transcript = self.recorder.record_and_transcribe(duration=5)
            if not transcript:
                self.speaker.speak("I didn't catch that. Could you repeat?")
                continue
            if transcript.lower().strip() in ("goodbye", "bye", "quit", "exit", "thank you"):
                self.speaker.speak("Goodbye! Have a great day.")
                break
            print(f"\nYou: {transcript}")
            answer = self.query(transcript)
            print(f"Assistant: {answer}\n")
            self.speaker.speak(answer)


# ── CLI ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AxiomCart Multi-Agent Voice System")
    parser.add_argument("--voice", action="store_true", help="Use microphone input + TTS output")
    parser.add_argument("--query", type=str, help="Run a single text query and exit")
    args = parser.parse_args()

    assistant = AxiomCartAssistant(enable_voice=args.voice)

    if args.query:
        print(assistant.query(args.query))
    elif args.voice:
        assistant.voice_loop()
    else:
        assistant.text_loop()


if __name__ == "__main__":
    main()
