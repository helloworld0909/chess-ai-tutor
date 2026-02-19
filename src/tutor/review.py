"""CLI entry point for Chess.com game review.

Fetches recent games for a username and opens a web UI to review them.
"""

from __future__ import annotations

import asyncio
import sys
import webbrowser
from pathlib import Path

import click
import uvicorn
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add src to path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()


async def _fetch_and_serve(
    username: str,
    months: int,
    port: int,
    stockfish_path: str | None,
    depth: int,
    no_browser: bool,
) -> None:
    """Fetch games and start the web server."""
    import os

    from tutor.chesscom import fetch_recent_games

    # Set stockfish path for web.py to pick up
    if stockfish_path:
        os.environ["STOCKFISH_PATH"] = stockfish_path

    # Fetch games
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            f"Fetching games for [bold cyan]{username}[/bold cyan]...", total=None
        )
        try:
            games = await fetch_recent_games(username, months=months)
            progress.update(task, completed=True)
        except Exception as e:
            progress.stop()
            console.print(f"[red]Error fetching games: {e}[/red]")
            raise SystemExit(1) from e

    if not games:
        console.print(
            f"[yellow]No games found for [bold]{username}[/bold] "
            f"in the last {months} month(s).[/yellow]"
        )
        raise SystemExit(0)

    console.print(
        f"[green]Found [bold]{len(games)}[/bold] game(s) for "
        f"[bold cyan]{username}[/bold cyan][/green]"
    )

    # Inject games + username into the web module
    import tutor.web as web_module

    web_module._games = games
    web_module._username = username

    url = f"http://localhost:{port}"
    console.print(f"\n[bold]Starting server at[/bold] [link={url}]{url}[/link]")
    console.print("Press [bold]Ctrl+C[/bold] to stop.\n")

    if not no_browser:
        # Open browser after a short delay to let server start
        loop = asyncio.get_event_loop()
        loop.call_later(1.0, lambda: webbrowser.open(url))

    # Start uvicorn
    config = uvicorn.Config(
        "tutor.web:app",
        host="0.0.0.0",
        port=port,
        log_level="error",  # suppress uvicorn noise
    )
    server = uvicorn.Server(config)
    await server.serve()


@click.command()
@click.argument("username")
@click.option(
    "--months",
    "-m",
    default=1,
    show_default=True,
    help="Number of recent months to fetch",
)
@click.option(
    "--port",
    "-p",
    default=8000,
    show_default=True,
    help="Port for the web server",
)
@click.option(
    "--stockfish",
    "-s",
    default=None,
    envvar="STOCKFISH_PATH",
    help="Path to Stockfish binary",
)
@click.option(
    "--depth",
    "-d",
    default=16,
    show_default=True,
    help="Stockfish analysis depth",
)
@click.option(
    "--no-browser",
    is_flag=True,
    default=False,
    help="Don't open a browser automatically",
)
def main(
    username: str,
    months: int,
    port: int,
    stockfish: str | None,
    depth: int,
    no_browser: bool,
) -> None:
    """Review your chess.com games with AI analysis.

    Fetches recent games for USERNAME from chess.com and opens
    a web UI to step through moves with Stockfish commentary.

    Example:

        chess-review magnus_carlsen

        chess-review your_username --months 3
    """
    console.print("\n[bold blue]Chess Game Review[/bold blue]")
    console.print(f"Player: [bold cyan]{username}[/bold cyan]\n")

    asyncio.run(
        _fetch_and_serve(
            username=username,
            months=months,
            port=port,
            stockfish_path=stockfish,
            depth=depth,
            no_browser=no_browser,
        )
    )


if __name__ == "__main__":
    main()
