"""MCP Server for Chess Analysis Tools."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Sequence

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    EmbeddedResource,
    TextContent,
    Tool,
)

from .stockfish import Stockfish
from .tools import ChessTools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global state
stockfish: Stockfish | None = None
chess_tools: ChessTools | None = None


@asynccontextmanager
async def lifespan(server: Server):
    """Manage Stockfish lifecycle."""
    global stockfish, chess_tools

    # Initialize Stockfish
    stockfish_path = os.environ.get("STOCKFISH_PATH", "stockfish")
    depth = int(os.environ.get("STOCKFISH_DEPTH", "20"))
    threads = int(os.environ.get("STOCKFISH_THREADS", "4"))
    hash_mb = int(os.environ.get("STOCKFISH_HASH_MB", "256"))

    logger.info(f"Starting Stockfish from: {stockfish_path}")
    stockfish = Stockfish(
        path=stockfish_path,
        depth=depth,
        threads=threads,
        hash_mb=hash_mb,
    )
    await stockfish.start()
    chess_tools = ChessTools(stockfish)
    logger.info("Chess MCP Server ready")

    try:
        yield
    finally:
        logger.info("Shutting down Stockfish")
        if stockfish:
            await stockfish.stop()


# Create server
app = Server("chess-mcp")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available chess tools."""
    return [
        Tool(
            name="get_best_move",
            description="Get the best move for a chess position. Returns the engine's recommended move with evaluation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "fen": {
                        "type": "string",
                        "description": "Position in FEN notation",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Search depth (default 20)",
                        "default": 20,
                    },
                },
                "required": ["fen"],
            },
        ),
        Tool(
            name="get_eval",
            description="Get the evaluation of a chess position. Returns score in centipawns and win probability.",
            inputSchema={
                "type": "object",
                "properties": {
                    "fen": {
                        "type": "string",
                        "description": "Position in FEN notation",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Search depth (default 20)",
                        "default": 20,
                    },
                },
                "required": ["fen"],
            },
        ),
        Tool(
            name="get_threats",
            description="Identify tactical threats in a position. Detects checks, hanging pieces, and tactical motifs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "fen": {
                        "type": "string",
                        "description": "Position in FEN notation",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Search depth (default 15)",
                        "default": 15,
                    },
                },
                "required": ["fen"],
            },
        ),
        Tool(
            name="compare_moves",
            description="Compare a user's move against the engine's best move. Returns classification (Best/Great/Good/Inaccuracy/Mistake/Blunder) and centipawn loss.",
            inputSchema={
                "type": "object",
                "properties": {
                    "fen": {
                        "type": "string",
                        "description": "Position in FEN notation",
                    },
                    "move": {
                        "type": "string",
                        "description": "User's move in UCI notation (e.g., 'e2e4')",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Search depth (default 20)",
                        "default": 20,
                    },
                },
                "required": ["fen", "move"],
            },
        ),
        Tool(
            name="get_legal_moves",
            description="Get all legal moves for a position, categorized into captures, checks, and other moves.",
            inputSchema={
                "type": "object",
                "properties": {
                    "fen": {
                        "type": "string",
                        "description": "Position in FEN notation",
                    },
                },
                "required": ["fen"],
            },
        ),
        Tool(
            name="validate_move",
            description="Validate if a move is legal and get the resulting position.",
            inputSchema={
                "type": "object",
                "properties": {
                    "fen": {
                        "type": "string",
                        "description": "Position in FEN notation",
                    },
                    "move": {
                        "type": "string",
                        "description": "Move in UCI notation (e.g., 'e2e4')",
                    },
                },
                "required": ["fen", "move"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(
    name: str, arguments: dict[str, Any]
) -> Sequence[TextContent | EmbeddedResource]:
    """Handle tool calls."""
    if chess_tools is None:
        return [TextContent(type="text", text="Error: Chess tools not initialized")]

    import json

    try:
        if name == "get_best_move":
            result = await chess_tools.get_best_move(
                fen=arguments["fen"],
                depth=arguments.get("depth", 20),
            )
        elif name == "get_eval":
            result = await chess_tools.get_eval(
                fen=arguments["fen"],
                depth=arguments.get("depth", 20),
            )
        elif name == "get_threats":
            result = await chess_tools.get_threats(
                fen=arguments["fen"],
                depth=arguments.get("depth", 15),
            )
        elif name == "compare_moves":
            result = await chess_tools.compare_moves(
                fen=arguments["fen"],
                move=arguments["move"],
                depth=arguments.get("depth", 20),
            )
        elif name == "get_legal_moves":
            result = await chess_tools.get_legal_moves(fen=arguments["fen"])
        elif name == "validate_move":
            result = await chess_tools.validate_move(
                fen=arguments["fen"],
                move=arguments["move"],
            )
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        if result.success:
            return [TextContent(type="text", text=json.dumps(result.data, indent=2))]
        else:
            return [TextContent(type="text", text=f"Error: {result.error}")]

    except Exception as e:
        logger.exception(f"Error in tool {name}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def run_server():
    """Run the MCP server."""
    async with lifespan(app):
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options(),
            )


def main():
    """Entry point for the MCP server."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
