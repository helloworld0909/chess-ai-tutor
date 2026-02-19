"""Tests for the FastAPI web server endpoints.

Uses TestClient without a context manager so the lifespan (Stockfish startup)
is skipped – we inject mocked state directly into the module globals.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tutor.web as web_module
from tutor.chesscom import Game, Move
from tutor.web import app

# ── Fixtures ──────────────────────────────────────────────────────────────────

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
AFTER_E4_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

SAMPLE_MOVES = [
    Move(0, "e4", "e2e4", STARTING_FEN, "white", 1),
    Move(1, "e5", "e7e5", AFTER_E4_FEN, "black", 1),
]

SAMPLE_GAME = Game(
    id="test-game-id",
    url="https://chess.com/game/1",
    white="alice",
    black="bob",
    result="1-0",
    result_detail="resignation",
    date="2024.01.15",
    time_control="600",
    moves=SAMPLE_MOVES,
    final_fen=AFTER_E4_FEN,
)


def _make_mock_stockfish(classification: str = "Best", cp_loss: int = 0) -> AsyncMock:
    """Build a mock Stockfish that returns controlled analysis results."""
    mock = AsyncMock()
    mock.compare_moves.return_value = {
        "classification": classification,
        "best_move": "e2e4",
        "cp_loss": cp_loss,
        "is_best": cp_loss == 0,
    }
    score = MagicMock()
    score.mate_in = None
    score.centipawns = 35
    line = MagicMock()
    line.best_move = "e2e4"
    line.score = score
    analysis = MagicMock()
    analysis.score = score
    analysis.lines = [line]
    mock.analyze.return_value = analysis
    mock.get_threats.return_value = {"threats": []}
    return mock


@pytest.fixture(autouse=True)
def inject_state():
    """Inject mock state before each test and restore afterwards."""
    original_games = web_module._games
    original_username = web_module._username
    original_stockfish = web_module._stockfish
    original_llm_client = web_module._llm_client
    original_llm_model = web_module._llm_model

    web_module._games = [SAMPLE_GAME]
    web_module._username = "alice"
    web_module._stockfish = _make_mock_stockfish()
    web_module._llm_client = None  # use template fallback by default
    web_module._llm_model = ""

    yield

    web_module._games = original_games
    web_module._username = original_username
    web_module._stockfish = original_stockfish
    web_module._llm_client = original_llm_client
    web_module._llm_model = original_llm_model


@pytest.fixture
def client() -> TestClient:
    # No context manager → lifespan skipped, we use injected mock state
    return TestClient(app, raise_server_exceptions=True)


# ── /api/username ─────────────────────────────────────────────────────────────


def test_get_username(client: TestClient):
    res = client.get("/api/username")
    assert res.status_code == 200
    assert res.json() == {"username": "alice"}


# ── /api/games ────────────────────────────────────────────────────────────────


def test_list_games_returns_summary(client: TestClient):
    res = client.get("/api/games")
    assert res.status_code == 200
    games = res.json()
    assert len(games) == 1
    g = games[0]
    assert g["id"] == "test-game-id"
    assert g["white"] == "alice"
    assert g["black"] == "bob"
    assert g["result"] == "1-0"
    assert g["move_count"] == 2


def test_list_games_title_from_perspective(client: TestClient):
    res = client.get("/api/games")
    title = res.json()[0]["title"]
    # alice is white and won, so title should mention Won
    assert "Won" in title
    assert "bob" in title


def test_list_games_empty(client: TestClient):
    web_module._games = []
    res = client.get("/api/games")
    assert res.status_code == 200
    assert res.json() == []


# ── /api/game/{id} ────────────────────────────────────────────────────────────


def test_get_game_returns_detail(client: TestClient):
    res = client.get("/api/game/test-game-id")
    assert res.status_code == 200
    data = res.json()
    assert data["id"] == "test-game-id"
    assert len(data["moves"]) == 2
    assert data["moves"][0]["san"] == "e4"
    assert data["moves"][0]["uci"] == "e2e4"
    assert data["moves"][0]["color"] == "white"
    assert data["final_fen"] == AFTER_E4_FEN


def test_get_game_not_found(client: TestClient):
    res = client.get("/api/game/nonexistent-id")
    assert res.status_code == 404


# ── /api/analyze ─────────────────────────────────────────────────────────────


def test_analyze_best_move(client: TestClient):
    res = client.post(
        "/api/analyze",
        json={
            "fen": STARTING_FEN,
            "move_uci": "e2e4",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["classification"] == "Best"
    assert data["is_best"] is True
    assert data["cp_loss"] == 0
    assert data["best_move"] == "e2e4"
    assert data["eval_before"] == "+0.35"
    assert len(data["comment"]) > 0
    assert data["comment_source"] == "template"


def test_analyze_blunder(client: TestClient):
    web_module._stockfish = _make_mock_stockfish("Blunder", cp_loss=300)
    res = client.post(
        "/api/analyze",
        json={
            "fen": STARTING_FEN,
            "move_uci": "f2f3",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["classification"] == "Blunder"
    assert data["is_best"] is False
    assert data["cp_loss"] == 300
    assert data["comment_source"] == "template"


def test_analyze_comment_not_empty(client: TestClient):
    res = client.post(
        "/api/analyze",
        json={
            "fen": STARTING_FEN,
            "move_uci": "e2e4",
        },
    )
    assert res.status_code == 200
    assert res.json()["comment"].strip() != ""


def test_analyze_stockfish_not_ready(client: TestClient):
    web_module._stockfish = None
    res = client.post(
        "/api/analyze",
        json={
            "fen": STARTING_FEN,
            "move_uci": "e2e4",
        },
    )
    assert res.status_code == 503


def test_analyze_eval_format_positive(client: TestClient):
    res = client.post(
        "/api/analyze",
        json={
            "fen": STARTING_FEN,
            "move_uci": "e2e4",
        },
    )
    # Positive eval starts with +
    assert res.json()["eval_before"].startswith("+")


def test_analyze_llm_comment_used(client: TestClient):
    """When LLM client is set and returns a response, comment_source is 'llm'."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "A solid central pawn move controlling key squares."
    mock_llm.chat.completions.create = AsyncMock(return_value=mock_response)

    web_module._llm_client = mock_llm
    web_module._llm_model = "test-model"

    res = client.post(
        "/api/analyze",
        json={"fen": STARTING_FEN, "move_uci": "e2e4"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["comment_source"] == "llm"
    assert "solid" in data["comment"]


def test_analyze_llm_fallback_on_error(client: TestClient):
    """When LLM raises an exception, falls back to template comment."""
    mock_llm = MagicMock()
    mock_llm.chat.completions.create = AsyncMock(side_effect=Exception("LLM unavailable"))

    web_module._llm_client = mock_llm
    web_module._llm_model = "test-model"

    res = client.post(
        "/api/analyze",
        json={"fen": STARTING_FEN, "move_uci": "e2e4"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["comment_source"] == "template"
    assert len(data["comment"]) > 0


def test_analyze_llm_strips_think_tags(client: TestClient):
    """Qwen3-style <think>...</think> blocks are stripped from the response."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = (
        "<think>Let me reason about this move carefully.</think>"
        "A strong opening move occupying the center."
    )
    mock_llm.chat.completions.create = AsyncMock(return_value=mock_response)

    web_module._llm_client = mock_llm
    web_module._llm_model = "test-model"

    res = client.post(
        "/api/analyze",
        json={"fen": STARTING_FEN, "move_uci": "e2e4"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "<think>" not in data["comment"]
    assert "strong opening" in data["comment"]
    assert data["comment_source"] == "llm"


# ── /api/board ────────────────────────────────────────────────────────────────


def test_get_board_svg_returns_svg(client: TestClient):
    res = client.get(f"/api/board?fen={STARTING_FEN}")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/svg+xml"
    assert res.text.startswith("<svg")


def test_get_board_svg_with_last_move(client: TestClient):
    res = client.get(f"/api/board?fen={AFTER_E4_FEN}&last_move=e2e4")
    assert res.status_code == 200
    assert "<svg" in res.text


def test_get_board_svg_flipped(client: TestClient):
    res = client.get(f"/api/board?fen={STARTING_FEN}&flip=true")
    assert res.status_code == 200
    assert "<svg" in res.text


def test_get_board_svg_invalid_fen(client: TestClient):
    res = client.get("/api/board?fen=not-a-valid-fen")
    assert res.status_code == 400


# ── Root ──────────────────────────────────────────────────────────────────────


def test_root_serves_html(client: TestClient):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]


# ── _board_ascii ──────────────────────────────────────────────────────────────


def test_board_ascii_has_file_labels():
    import chess

    from tutor.web import _board_ascii

    result = _board_ascii(chess.Board())
    assert "a b c d e f g h" in result


def test_board_ascii_has_rank_labels():
    import chess

    from tutor.web import _board_ascii

    result = _board_ascii(chess.Board())
    for rank in "12345678":
        assert rank in result


def test_board_ascii_white_to_move():
    import chess

    from tutor.web import _board_ascii

    result = _board_ascii(chess.Board())
    assert "White to move" in result


def test_board_ascii_black_to_move():
    import chess

    from tutor.web import _board_ascii

    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    result = _board_ascii(board)
    assert "Black to move" in result


# ── _move_facts ───────────────────────────────────────────────────────────────


def test_move_facts_capture():
    import chess

    from tutor.web import _move_facts

    # 1.e4 d5 — white can capture exd5
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
    facts = _move_facts(board, chess.Move.from_uci("e4d5"))
    assert any("captures pawn" in f for f in facts)


def test_move_facts_check():
    import chess

    from tutor.web import _move_facts

    # White rook on a1, black king on h8, white king on h1 — Ra8 gives check along rank 8
    board = chess.Board("7k/8/8/8/8/8/8/R6K w - - 0 1")
    facts = _move_facts(board, chess.Move.from_uci("a1a8"))
    assert any("check" in f for f in facts)


def test_move_facts_castling():
    import chess

    from tutor.web import _move_facts

    # Standard kingside castling position
    board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    facts = _move_facts(board, chess.Move.from_uci("e1g1"))
    assert any("castles" in f for f in facts)


def test_move_facts_returns_list_for_quiet_move():
    import chess

    from tutor.web import _move_facts

    facts = _move_facts(chess.Board(), chess.Move.from_uci("e2e4"))
    assert isinstance(facts, list)


def test_move_facts_no_facts_for_missing_piece():
    import chess

    from tutor.web import _move_facts

    # Empty board — no piece on e2, should return []
    board = chess.Board("8/8/8/8/8/8/8/K6k w - - 0 1")
    facts = _move_facts(board, chess.Move.from_uci("e2e4"))
    assert facts == []


# ── board_ascii in LLM prompt ─────────────────────────────────────────────────


def test_analyze_llm_prompt_includes_board(client: TestClient):
    """LLM receives a user message containing the ASCII board."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "A fine central pawn move."
    mock_llm.chat.completions.create = AsyncMock(return_value=mock_response)

    web_module._llm_client = mock_llm
    web_module._llm_model = "test-model"

    client.post("/api/analyze", json={"fen": STARTING_FEN, "move_uci": "e2e4"})

    assert mock_llm.chat.completions.create.called
    messages = mock_llm.chat.completions.create.call_args.kwargs["messages"]
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    assert "a b c d e f g h" in user_content


def test_analyze_llm_prompt_includes_move_facts(client: TestClient):
    """LLM user message includes the verified move facts section."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Good."
    mock_llm.chat.completions.create = AsyncMock(return_value=mock_response)

    web_module._llm_client = mock_llm
    web_module._llm_model = "test-model"

    # exd5 — a capture, so move facts will be non-empty
    fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
    client.post("/api/analyze", json={"fen": fen, "move_uci": "e4d5"})

    messages = mock_llm.chat.completions.create.call_args.kwargs["messages"]
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    assert "Verified move facts" in user_content
