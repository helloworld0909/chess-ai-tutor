"""Simple LLM agent with chess tool calling.

Uses OpenAI-compatible API (vLLM) with function calling
to create a chess tutor that can analyze positions.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_mcp.stockfish import Stockfish
from tutor.llm_tools import CHESS_TOOLS, ChessToolHandler


SYSTEM_PROMPT = """\
You are a friendly chess tutor helping a student improve their game.

When analyzing positions or moves:
1. Use the available tools to get accurate engine analysis
2. Explain concepts in clear, educational language
3. Focus on the "why" - explain strategic and tactical ideas
4. Be encouraging but honest about mistakes
5. Suggest improvements when moves are suboptimal

You have access to chess analysis tools:
- get_best_move: Find the engine's recommended move
- get_eval: Get position evaluation
- analyze_move: Classify a move (Best/Great/Good/Inaccuracy/Mistake/Blunder)
- get_legal_moves: List all legal moves
- validate_move: Check if a move is legal

Always use tools to verify your analysis rather than guessing.
Respond in a conversational, helpful tone like a patient coach.
"""


class ChessAgent:
    """Simple chess tutoring agent with tool calling."""

    def __init__(
        self,
        base_url: str = "http://localhost:9000/v1",
        api_key: str = "dummy",
        model: str = "qwen3-vl-30b-a3b",
        stockfish_path: str | None = None,
        max_tool_rounds: int = 5,
    ):
        """Initialize the chess agent.

        Args:
            base_url: vLLM OpenAI-compatible API URL
            api_key: API key (can be dummy for local vLLM)
            model: Model name to use
            stockfish_path: Path to Stockfish binary
            max_tool_rounds: Maximum rounds of tool calling
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_tool_rounds = max_tool_rounds

        self.stockfish = Stockfish(path=stockfish_path)
        self.tool_handler: ChessToolHandler | None = None

        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    async def start(self) -> None:
        """Start the agent (initialize Stockfish)."""
        await self.stockfish.start()
        self.tool_handler = ChessToolHandler(self.stockfish)

    async def stop(self) -> None:
        """Stop the agent (cleanup Stockfish)."""
        await self.stockfish.stop()

    async def __aenter__(self) -> "ChessAgent":
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()

    async def chat(self, user_message: str) -> str:
        """Send a message and get a response.

        Handles tool calling automatically in a loop.

        Args:
            user_message: The user's message

        Returns:
            The assistant's final response
        """
        if self.tool_handler is None:
            raise RuntimeError("Agent not started. Call start() first.")

        # Add user message
        self.messages.append({"role": "user", "content": user_message})

        # Tool calling loop
        for _ in range(self.max_tool_rounds):
            # Call the model
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,  # type: ignore[arg-type]
                tools=CHESS_TOOLS,  # type: ignore[arg-type]
                tool_choice="auto",
            )

            assistant_message = response.choices[0].message

            # Add assistant message to history
            self.messages.append(assistant_message.model_dump())

            # Check if we need to call tools
            if not assistant_message.tool_calls:
                # No tool calls, return the response
                return assistant_message.content or ""

            # Execute tool calls
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name  # type: ignore[union-attr]
                function_args = json.loads(tool_call.function.arguments)  # type: ignore[union-attr]

                # Execute the tool
                result = await self.tool_handler.handle_tool_call(
                    function_name, function_args
                )

                # Add tool result to messages
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        # Max rounds reached, get final response without tools
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,  # type: ignore[arg-type]
        )

        return response.choices[0].message.content or ""

    def reset(self) -> None:
        """Reset conversation history."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]


async def main():
    """Demo the chess agent."""
    import argparse

    parser = argparse.ArgumentParser(description="Chess AI Tutor Agent")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LLM_BASE_URL", "http://localhost:9000/v1"),
        help="vLLM API base URL",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LLM_MODEL", "qwen3-vl-30b-a3b"),
        help="Model name",
    )
    parser.add_argument(
        "--stockfish",
        default=os.environ.get("STOCKFISH_PATH"),
        help="Path to Stockfish binary",
    )
    args = parser.parse_args()

    print("Chess AI Tutor")
    print("=" * 40)
    print(f"LLM: {args.model}")
    print(f"API: {args.base_url}")
    print("=" * 40)
    print("Type 'quit' to exit, 'reset' to clear history\n")

    async with ChessAgent(
        base_url=args.base_url,
        model=args.model,
        stockfish_path=args.stockfish,
    ) as agent:
        while True:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit", "q"]:
                    break

                if user_input.lower() == "reset":
                    agent.reset()
                    print("Conversation reset.")
                    continue

                print("\nTutor: ", end="", flush=True)
                response = await agent.chat(user_input)
                print(response)

            except KeyboardInterrupt:
                print("\nUse 'quit' to exit.")
            except Exception as e:
                print(f"\nError: {e}")

    print("\nGoodbye!")


if __name__ == "__main__":
    asyncio.run(main())
