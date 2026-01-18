"""CLI interface for Chess Tutor.

Interactive command-line chess analysis and tutoring.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import chess
import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_mcp.stockfish import Stockfish
from chess_mcp.representations import fen_to_ascii
from verification.legality import parse_move_flexible


console = Console()

# Starting position FEN
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class TutorSession:
    """Interactive chess tutoring session."""

    def __init__(
        self,
        stockfish_path: str | None = None,
        stockfish_depth: int = 20,
    ):
        self.stockfish = Stockfish(
            path=stockfish_path,
            depth=stockfish_depth,
        )
        self.board = chess.Board()
        self.move_history: list[str] = []
        self.running = False

    async def start(self):
        """Start the session."""
        await self.stockfish.start()
        self.running = True

    async def stop(self):
        """Stop the session."""
        await self.stockfish.stop()
        self.running = False

    def display_board(self):
        """Display the current board position."""
        ascii_board = fen_to_ascii(self.board.fen())

        # Add move counter and side to move
        side = "White" if self.board.turn == chess.WHITE else "Black"
        move_num = self.board.fullmove_number

        console.print()
        console.print(Panel(
            ascii_board,
            title=f"Move {move_num} - {side} to play",
            border_style="blue",
        ))
        console.print()

    def display_position_info(self):
        """Display detailed position information."""
        table = Table(title="Position Info")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        # FEN
        table.add_row("FEN", self.board.fen())

        # Castling
        castling = []
        if self.board.has_kingside_castling_rights(chess.WHITE):
            castling.append("O-O (White)")
        if self.board.has_queenside_castling_rights(chess.WHITE):
            castling.append("O-O-O (White)")
        if self.board.has_kingside_castling_rights(chess.BLACK):
            castling.append("O-O (Black)")
        if self.board.has_queenside_castling_rights(chess.BLACK):
            castling.append("O-O-O (Black)")
        table.add_row("Castling", ", ".join(castling) or "None")

        # Check status
        if self.board.is_checkmate():
            table.add_row("Status", "[red]Checkmate![/red]")
        elif self.board.is_stalemate():
            table.add_row("Status", "[yellow]Stalemate[/yellow]")
        elif self.board.is_check():
            table.add_row("Status", "[yellow]Check[/yellow]")
        else:
            table.add_row("Status", "Normal")

        # Legal moves count
        table.add_row("Legal moves", str(len(list(self.board.legal_moves))))

        console.print(table)

    async def analyze_position(self, depth: int | None = None):
        """Analyze the current position with Stockfish."""
        with console.status("Analyzing position..."):
            analysis = await self.stockfish.analyze(
                self.board.fen(),
                depth=depth,
                multipv=3,
            )

        console.print("\n[bold]Engine Analysis[/bold]")

        for i, line in enumerate(analysis.lines, 1):
            score = line.score
            pv = " ".join(line.pv[:5])

            if score.mate_in is not None:
                score_str = f"Mate in {abs(score.mate_in)}"
            else:
                score_str = f"{score.value / 100:+.2f}"

            console.print(f"  {i}. [cyan]{line.pv[0]}[/cyan] ({score_str}): {pv}")

        console.print()

    async def analyze_move(self, move: str):
        """Analyze a specific move."""
        # Validate and parse move
        result = parse_move_flexible(self.board.fen(), move)

        if not result.valid:
            console.print(f"[red]Error: {result.error}[/red]")
            return

        # Type narrowing: move_uci and move_san are guaranteed non-None when valid
        assert result.move_uci is not None
        assert result.move_san is not None

        with console.status("Analyzing move..."):
            comparison = await self.stockfish.compare_moves(
                self.board.fen(),
                result.move_uci,
            )

        if "error" in comparison:
            console.print(f"[red]Error: {comparison['error']}[/red]")
            return

        # Display analysis
        classification = comparison["classification"]
        cp_loss = comparison["cp_loss"]
        best_move = comparison["best_move"]
        is_best = comparison["is_best"]

        # Color based on classification
        color_map = {
            "Best": "green",
            "Great": "green",
            "Good": "yellow",
            "Inaccuracy": "yellow",
            "Mistake": "red",
            "Blunder": "red",
        }
        color = color_map.get(classification, "white")

        console.print(f"\n[bold]Move Analysis: {result.move_san}[/bold]")
        console.print(f"Classification: [{color}]{classification}[/{color}]")

        if not is_best:
            console.print(f"Best move: [cyan]{best_move}[/cyan]")
            if cp_loss > 0:
                console.print(f"Centipawn loss: {cp_loss}")

        # Provide a human-like explanation
        explanation = self._generate_explanation(
            result.move_san,
            classification,
            best_move,
            is_best,
        )
        console.print(f"\n{explanation}")

    def _generate_explanation(
        self,
        move: str,
        classification: str,
        best_move: str,
        is_best: bool,
    ) -> str:
        """Generate a human-like explanation for the move."""
        if is_best:
            return (
                f"Excellent choice! {move} is the engine's top recommendation. "
                "This move optimally addresses the demands of the position."
            )

        if classification in ["Great", "Good"]:
            return (
                f"{move} is a reasonable move that keeps the position balanced. "
                f"The engine slightly prefers {best_move}, but your move is perfectly playable."
            )

        if classification == "Inaccuracy":
            return (
                f"{move} is slightly imprecise. While not a serious error, "
                f"consider {best_move} which better addresses the position's needs."
            )

        if classification == "Mistake":
            return (
                f"{move} is a mistake that gives your opponent an advantage. "
                f"The correct move was {best_move}. Let's look at why your move was problematic..."
            )

        if classification == "Blunder":
            return (
                f"{move} is a significant error! This move seriously compromises your position. "
                f"You should have played {best_move} instead. "
                "Always check for tactical threats before committing to a move."
            )

        return f"Analysis: {classification}"

    async def make_move(self, move: str):
        """Make a move on the board."""
        result = parse_move_flexible(self.board.fen(), move)

        if not result.valid:
            console.print(f"[red]Error: {result.error}[/red]")
            return False

        # Type narrowing: move_uci is guaranteed non-None when valid
        assert result.move_uci is not None

        # Apply move
        chess_move = chess.Move.from_uci(result.move_uci)
        san = self.board.san(chess_move)
        self.board.push(chess_move)
        self.move_history.append(san)

        console.print(f"[green]Played: {san}[/green]")

        # Check game state
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            console.print(f"\n[bold red]Checkmate! {winner} wins![/bold red]")
        elif self.board.is_stalemate():
            console.print("\n[bold yellow]Stalemate! The game is drawn.[/bold yellow]")
        elif self.board.is_check():
            console.print("[yellow]Check![/yellow]")

        return True

    def undo_move(self):
        """Undo the last move."""
        if self.move_history:
            self.board.pop()
            undone = self.move_history.pop()
            console.print(f"[yellow]Undone: {undone}[/yellow]")
        else:
            console.print("[red]No moves to undo[/red]")

    def new_game(self, fen: str | None = None):
        """Start a new game."""
        if fen:
            try:
                self.board = chess.Board(fen)
            except ValueError as e:
                console.print(f"[red]Invalid FEN: {e}[/red]")
                return
        else:
            self.board = chess.Board()

        self.move_history = []
        console.print("[green]New game started![/green]")

    def show_moves(self):
        """Show legal moves."""
        moves = sorted([self.board.san(m) for m in self.board.legal_moves])

        # Group by piece
        console.print("\n[bold]Legal Moves[/bold]")

        # Categorize
        pawn_moves = [m for m in moves if m[0].islower() or m[0] in "abcdefgh"]
        piece_moves = [m for m in moves if m[0] in "NBRQK"]
        castling = [m for m in moves if m in ["O-O", "O-O-O"]]

        if pawn_moves:
            console.print(f"  Pawns: {', '.join(pawn_moves)}")
        if piece_moves:
            console.print(f"  Pieces: {', '.join(piece_moves)}")
        if castling:
            console.print(f"  Castling: {', '.join(castling)}")

        console.print()

    def show_help(self):
        """Show help information."""
        help_text = """
## Chess Tutor Commands

| Command | Description |
|---------|-------------|
| `<move>` | Make a move (e.g., `e4`, `Nf3`, `O-O`) |
| `analyze [move]` | Analyze position or specific move |
| `undo` | Undo last move |
| `moves` | Show legal moves |
| `info` | Show position details |
| `new [fen]` | Start new game |
| `fen` | Show current FEN |
| `pgn` | Show game PGN |
| `help` | Show this help |
| `quit` | Exit the tutor |

## Tips
- Enter moves in standard chess notation (SAN)
- Use `analyze` to get engine recommendations
- Use `analyze <move>` to evaluate a specific move before playing it
        """
        console.print(Markdown(help_text))


async def run_interactive(
    stockfish_path: str | None = None,
    depth: int = 20,
    fen: str | None = None,
):
    """Run interactive CLI session."""
    session = TutorSession(stockfish_path=stockfish_path, stockfish_depth=depth)

    try:
        await session.start()

        console.print("\n[bold blue]Chess Tutor[/bold blue]")
        console.print("Type 'help' for commands, 'quit' to exit.\n")

        if fen:
            session.new_game(fen)

        session.display_board()

        while session.running:
            try:
                # Get input
                prompt = f"[{'white' if session.board.turn == chess.WHITE else 'black'}]> "
                cmd = Prompt.ask(prompt).strip()

                if not cmd:
                    continue

                # Parse command
                parts = cmd.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                if command in ["quit", "exit", "q"]:
                    break

                elif command == "help":
                    session.show_help()

                elif command in ["board", "show"]:
                    session.display_board()

                elif command == "info":
                    session.display_position_info()

                elif command == "analyze":
                    if args:
                        await session.analyze_move(args)
                    else:
                        await session.analyze_position()

                elif command == "moves":
                    session.show_moves()

                elif command == "undo":
                    session.undo_move()
                    session.display_board()

                elif command == "new":
                    session.new_game(args if args else None)
                    session.display_board()

                elif command == "fen":
                    console.print(session.board.fen())

                elif command == "pgn":
                    if session.move_history:
                        moves = []
                        for i, m in enumerate(session.move_history):
                            if i % 2 == 0:
                                moves.append(f"{i // 2 + 1}. {m}")
                            else:
                                moves.append(m)
                        console.print(" ".join(moves))
                    else:
                        console.print("No moves played yet.")

                else:
                    # Try to make a move
                    if await session.make_move(cmd):
                        session.display_board()

            except KeyboardInterrupt:
                console.print("\nUse 'quit' to exit.")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    finally:
        await session.stop()

    console.print("\n[blue]Thanks for using Chess Tutor![/blue]\n")


@click.command()
@click.option(
    "--stockfish",
    "-s",
    default=None,
    help="Path to Stockfish binary",
)
@click.option(
    "--depth",
    "-d",
    default=20,
    help="Analysis depth",
)
@click.option(
    "--fen",
    "-f",
    default=None,
    help="Starting position FEN",
)
def main(stockfish: str | None, depth: int, fen: str | None):
    """Chess Tutor - Interactive chess analysis and learning."""
    asyncio.run(run_interactive(
        stockfish_path=stockfish,
        depth=depth,
        fen=fen,
    ))


if __name__ == "__main__":
    main()
