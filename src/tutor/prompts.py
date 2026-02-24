"""Shared prompt constants and utilities for chess coaching.

Used by both the inference server (web.py) and the training data pipeline
(prepare_datasets.py) to guarantee training ↔ inference format alignment.
"""

from __future__ import annotations

import chess

# ---------------------------------------------------------------------------
# System prompt — used verbatim at both training and inference time
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert chess coach giving move-by-move feedback.\n\n"
    "Core rules:\n"
    "- Your comment MUST agree with the engine classification. "
    "Best/Great → explain why it is strong. "
    "Good → note it is solid. "
    "Inaccuracy → explain the missed opportunity. "
    "Mistake/Blunder → explain the error and the correct idea clearly.\n"
    "- Never say a 'Best' move is suboptimal.\n"
    "- Do not parrot the classification label or the eval number.\n"
    "- The verified move facts are ground truth — do not contradict them.\n\n"
    "Depth by game phase:\n"
    "OPENING (moves 1–12, most pieces on board): Be concise. Name the opening "
    "or variation (e.g. 'Ruy Lopez Berlin Defense', 'King's Indian Attack'). "
    "State whether the move is recognized opening theory or a deviation. "
    "Use web_search to verify the exact opening name or look up theoretical lines "
    "if you are not certain. Skip deep engine analysis — opening knowledge trumps "
    "engine lines here. 2–3 sentences max.\n\n"
    "MIDDLEGAME: Explain the key tactical or strategic idea. Reference the "
    "engine's top candidates when relevant. Use analyze_position to verify "
    "critical variations before asserting concrete lines. 3–5 sentences.\n\n"
    "ENDGAME: Analyze in depth. Use analyze_position to explore the key lines "
    "before writing. Focus on king activity, pawn advancement, piece "
    "coordination, and conversion technique. Cite specific variations. "
    "4–6 sentences.\n\n"
    "Calibrate length to complexity — a simple recapture needs 2 sentences, "
    "a deep positional sacrifice needs 6.\n\n"
    "OUTPUT FORMAT: After your reasoning, write your final coaching comment "
    "wrapped in <comment>...</comment> tags. "
    "Example: <comment>This move centralises the knight and eyes the weak d5 square.</comment> "
    "The text inside <comment> is the only part shown to the student — keep it "
    "clean, direct coaching language with no meta-commentary."
)


# ---------------------------------------------------------------------------
# Board / move utilities
# ---------------------------------------------------------------------------


def board_ascii(board: chess.Board) -> str:
    """Return a labelled ASCII board (rank/file labels) for LLM context."""
    rows = str(board).split("\n")  # rank 8 at top
    lines = ["  a b c d e f g h"]
    for i, row in enumerate(rows):
        lines.append(f"{8 - i} {row}")
    lines.append(f"  ({'White' if board.turn == chess.WHITE else 'Black'} to move)")
    return "\n".join(lines)


def move_facts(board: chess.Board, move: chess.Move) -> list[str]:
    """Extract verifiable mechanical facts about a move using python-chess."""
    piece = board.piece_at(move.from_square)
    if piece is None:
        return []

    facts: list[str] = []
    piece_name = chess.piece_name(piece.piece_type)
    our_color = piece.color
    their_color = not our_color
    to_sq = chess.square_name(move.to_square)

    # Special moves
    if board.is_castling(move):
        facts.append("king castles for safety and connects the rooks")
        return facts
    if board.is_en_passant(move):
        facts.append(f"en passant capture on {to_sq}")
    elif board.is_capture(move):
        cap = board.piece_at(move.to_square)
        if cap:
            facts.append(f"captures {chess.piece_name(cap.piece_type)} on {to_sq}")

    # Check
    if board.gives_check(move):
        facts.append("gives check to the opponent's king")

    # Was the moving piece under attack before? (defensive retreat)
    if board.attackers(their_color, move.from_square):
        facts.append(f"rescues the {piece_name} which was under attack")

    # Analyse the resulting position
    board_after = board.copy()
    board_after.push(move)

    attacked_sq = board_after.attacks(move.to_square)

    # Enemy pieces now attacked
    attacked_enemies = [
        f"{chess.piece_name(p.piece_type)} on {chess.square_name(sq)}"
        for sq in attacked_sq
        if (p := board_after.piece_at(sq)) and p.color == their_color
    ]
    if attacked_enemies:
        facts.append(f"now attacks {', '.join(attacked_enemies)}")

    # Own pieces now supported by the moved piece
    supported = [
        f"{chess.piece_name(p.piece_type)} on {chess.square_name(sq)}"
        for sq in attacked_sq
        if (p := board_after.piece_at(sq)) and p.color == our_color and sq != move.to_square
    ]
    if supported:
        facts.append(f"defends own {', '.join(supported)}")

    # Own pieces that lost their defender after this piece moved away
    now_hanging = [
        f"{chess.piece_name(p.piece_type)} on {chess.square_name(sq)}"
        for sq in board.attacks(move.from_square)
        if (p := board.piece_at(sq))
        and p.color == our_color
        and sq != move.to_square
        and not board_after.is_attacked_by(our_color, sq)  # no longer defended
        and board_after.is_attacked_by(their_color, sq)  # opponent attacks it
    ]
    if now_hanging:
        facts.append(f"leaves own {', '.join(now_hanging)} undefended")

    return facts


# ---------------------------------------------------------------------------
# Textbook rewrite system prompt
# ---------------------------------------------------------------------------

TEXTBOOK_SYSTEM_PROMPT = (
    "You are an expert chess coach reformatting a human expert's game annotation "
    "into a standard coaching comment.\n\n"
    "Your ONLY task is to rewrite the provided expert annotation in clear coaching language. "
    "The expert annotation is ground truth — it comes from a qualified chess instructor "
    "and takes priority over engine classifications.\n\n"
    "You MUST rewrite to assume that the game is currently played by your student, "
    "even if the expert annotation says the game is played by named players. "
    "Do NOT mention the name of the original player like Akopian or Atkins.\n\n"
    "Rules:\n"
    "- PRESERVE every chess insight, claim, plan, and explanation from the expert annotation. "
    "Do not drop, weaken, or contradict any point the expert makes.\n"
    "- Do NOT introduce new chess claims or analysis beyond what the expert wrote or "
    "what Stockfish directly supports.\n"
    "- Rewrite in direct, concise coaching language (3–6 sentences).\n"
    "- If the expert's wording is already clear and well-structured, a light rewrite is fine; "
    "do not pad or dilute it.\n\n"
    "FILTERING RULE — respond with exactly the word SKIP (nothing else) when the annotation:\n"
    "- Contains no chess instruction (e.g. 'Forced.', 'Nice!', 'Good move.', '0-1')\n"
    "- Is purely a game-result or time-forfeit note\n"
    "- Is only a quiz/question with no answer or explanation\n"
    "- Is too vague to convey any chess idea (e.g. 'Black continues their plan.')\n\n"
    "OUTPUT FORMAT: Write your final coaching comment wrapped in <comment>...</comment> tags. "
    "Example: <comment>The knight leaps to its outpost on d5, eyeing the weak c7-pawn.</comment> "
    "The text inside <comment> is the only part shown to the student. "
    "For SKIP responses, reply with exactly: <comment>SKIP</comment>"
)

# Few-shot examples inserted into the message list before the real query.
# Each tuple is (user_content, assistant_content).
TEXTBOOK_FEW_SHOT: list[tuple[str, str]] = [
    # --- Negative: pure result ---
    (
        "Move played: Rd1\nClassification: best\nEngine evaluation before move: +3.2\n\n"
        "Expert annotation to preserve and reformat:\n0-1 White resigns.\n\n"
        "Rewrite the expert annotation above into coaching style. "
        "Every insight from the expert must appear in your response. "
        "Do not omit or contradict any point.",
        "<comment>SKIP</comment>",
    ),
    # --- Negative: single-word / no insight ---
    (
        "Move played: Nf3\nClassification: best\nEngine evaluation before move: +0.1\n\n"
        "Expert annotation to preserve and reformat:\nForced.\n\n"
        "Rewrite the expert annotation above into coaching style. "
        "Every insight from the expert must appear in your response. "
        "Do not omit or contradict any point.",
        "<comment>SKIP</comment>",
    ),
    # --- Negative: quiz with no answer ---
    (
        "Move played: e4\nClassification: best\nEngine evaluation before move: +0.3\n\n"
        "Expert annotation to preserve and reformat:\n"
        "What would you play here? Find the best continuation.\n\n"
        "Rewrite the expert annotation above into coaching style. "
        "Every insight from the expert must appear in your response. "
        "Do not omit or contradict any point.",
        "<comment>SKIP</comment>",
    ),
    # --- Negative: vague with no chess idea ---
    (
        "Move played: Rb8\nClassification: good\nEngine evaluation before move: -0.5\n\n"
        "Expert annotation to preserve and reformat:\nBlack continues their plan.\n\n"
        "Rewrite the expert annotation above into coaching style. "
        "Every insight from the expert must appear in your response. "
        "Do not omit or contradict any point.",
        "<comment>SKIP</comment>",
    ),
    # --- Positive: genuine instructive annotation ---
    (
        "Move played: Nd5\nClassification: best\nEngine evaluation before move: +1.8\n\n"
        "Expert annotation to preserve and reformat:\n"
        "The knight leaps to its dream square. From d5 it cannot be challenged by a pawn, "
        "attacks the c7 weakness, and eyes the f6 entry — all at once. "
        "This is the outpost that the entire opening was played for.\n\n"
        "Rewrite the expert annotation above into coaching style. "
        "Every insight from the expert must appear in your response. "
        "Do not omit or contradict any point.",
        "<comment>The knight jumps to d5 — a permanent outpost that no enemy pawn can challenge. "
        "From this dominant square it simultaneously pressures the weak c7-pawn and "
        "threatens to penetrate via f6, combining attack and restriction in one move. "
        "The entire opening strategy has been building toward this moment.</comment>",
    ),
]


def format_textbook_prompt(
    board_ascii_str: str,
    san: str,
    classification: str,
    eval_str: str,
    expert_annotation: str,
    facts: list[str] | None = None,
    fen: str = "",
) -> str:
    """Build the user message for textbook annotation rewriting.

    The expert annotation is the primary content — the LLM's job is to reformat
    it faithfully, not to generate fresh analysis.
    """
    facts_line = (
        "Verified move facts:\n" + "\n".join(f"- {f}" for f in facts) + "\n" if facts else ""
    )
    fen_line = f"FEN: {fen}\n" if fen else ""
    board_section = (
        f"Board position before the move:\n{board_ascii_str}\n{fen_line}\n"
        if board_ascii_str
        else ""
    )

    return (
        f"{board_section}"
        f"Move played: {san}\n"
        f"Classification: {classification}\n"
        f"Engine evaluation before move: {eval_str}\n"
        f"{facts_line}"
        f"\nExpert annotation to preserve and reformat:\n{expert_annotation}\n"
        "\nRewrite the expert annotation above into coaching style. "
        "Every insight from the expert must appear in your response. "
        "Do not omit or contradict any point."
    )


# ---------------------------------------------------------------------------
# Prompt formatting — shared between training and inference
# ---------------------------------------------------------------------------


def format_user_prompt(
    board_ascii_str: str,
    san: str,
    classification: str,
    eval_str: str,
    best_move: str = "",
    cp_loss: int = 0,
    candidates: list[str] | None = None,
    opponent_threats: list[str] | None = None,
    facts: list[str] | None = None,
    fen: str = "",
    cct: dict[str, list[str]] | None = None,
) -> str:
    """Build the user message for the chess coaching prompt.

    This function is the single source of truth for the user prompt format,
    used identically by the inference server and the training data pipeline.
    """
    best_line = (
        f"Engine's best move was: {best_move} (−{cp_loss} centipawns)\n" if cp_loss > 0 else ""
    )
    candidates_line = f"Engine's top candidates: {', '.join(candidates)}\n" if candidates else ""
    threats_line = (
        f"Opponent's threats if you passed: {', '.join(opponent_threats)}\n"
        if opponent_threats
        else ""
    )
    facts_line = (
        "Verified move facts:\n" + "\n".join(f"- {f}" for f in facts) + "\n" if facts else ""
    )
    fen_line = f"FEN: {fen}\n" if fen else ""
    board_section = (
        f"Board position before the move:\n{board_ascii_str}\n{fen_line}\n"
        if board_ascii_str
        else ""
    )
    cct_line = ""
    if cct:
        parts = []
        if cct.get("checks"):
            parts.append(f"Checks: {', '.join(cct['checks'])}")
        if cct.get("captures"):
            parts.append(f"Captures: {', '.join(cct['captures'])}")
        if cct.get("threats"):
            parts.append(f"Threats: {', '.join(cct['threats'])}")
        if parts:
            cct_line = "Tactical options (CCT): " + " | ".join(parts) + "\n"

    return (
        f"{board_section}"
        f"Move played: {san}\n"
        f"Classification: {classification}\n"
        f"Engine evaluation before move: {eval_str}\n"
        f"{best_line}"
        f"{candidates_line}"
        f"{threats_line}"
        f"{cct_line}"
        f"{facts_line}"
        "\nExplain the chess idea behind this move in 4-6 sentences. "
        "Cover the immediate tactical or positional purpose, the strategic plan it supports, "
        "and any relevant opening theory, piece activity, pawn structure, or king safety considerations. "
        "If verified move facts are listed, do not contradict them."
    )
