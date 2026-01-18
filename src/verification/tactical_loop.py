"""Tactical verification loop for LLM output validation.

Cross-checks LLM move classifications against Stockfish engine
and rejects/regenerates responses with incorrect assessments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Awaitable, TYPE_CHECKING

if TYPE_CHECKING:
    from chess_mcp.stockfish import Stockfish


class MoveClassification(Enum):
    """Move quality classifications."""

    BEST = "Best"
    GREAT = "Great"
    GOOD = "Good"
    INACCURACY = "Inaccuracy"
    MISTAKE = "Mistake"
    BLUNDER = "Blunder"
    UNKNOWN = "Unknown"


# Centipawn loss thresholds for each classification
CLASSIFICATION_THRESHOLDS = {
    MoveClassification.BEST: 10,       # Within 0.1 pawn of best
    MoveClassification.GREAT: 30,      # Within 0.3 pawn
    MoveClassification.GOOD: 80,       # Minor inaccuracy
    MoveClassification.INACCURACY: 150,
    MoveClassification.MISTAKE: 300,
    # BLUNDER: anything worse
}


@dataclass
class VerificationResult:
    """Result of verifying an LLM response."""

    valid: bool
    llm_classification: MoveClassification
    engine_classification: MoveClassification
    centipawn_loss: int
    reason: str | None = None
    correction: str | None = None


def classify_move_by_cp_loss(cp_loss: int) -> MoveClassification:
    """Classify a move based on centipawn loss.

    Args:
        cp_loss: Centipawn loss (positive = worse than best)

    Returns:
        MoveClassification enum
    """
    if cp_loss <= CLASSIFICATION_THRESHOLDS[MoveClassification.BEST]:
        return MoveClassification.BEST
    elif cp_loss <= CLASSIFICATION_THRESHOLDS[MoveClassification.GREAT]:
        return MoveClassification.GREAT
    elif cp_loss <= CLASSIFICATION_THRESHOLDS[MoveClassification.GOOD]:
        return MoveClassification.GOOD
    elif cp_loss <= CLASSIFICATION_THRESHOLDS[MoveClassification.INACCURACY]:
        return MoveClassification.INACCURACY
    elif cp_loss <= CLASSIFICATION_THRESHOLDS[MoveClassification.MISTAKE]:
        return MoveClassification.MISTAKE
    else:
        return MoveClassification.BLUNDER


def extract_classification_from_text(text: str) -> MoveClassification:
    """Extract move classification from LLM response text.

    Args:
        text: LLM response text

    Returns:
        Extracted classification or UNKNOWN
    """
    text_lower = text.lower()

    # Check for explicit classifications
    patterns = [
        (r"\bblunder\b", MoveClassification.BLUNDER),
        (r"\bmistake\b", MoveClassification.MISTAKE),
        (r"\binaccuracy\b", MoveClassification.INACCURACY),
        (r"\bgood move\b|\bgood\b", MoveClassification.GOOD),
        (r"\bgreat move\b|\bgreat\b|\bexcellent\b|\bstrong\b", MoveClassification.GREAT),
        (r"\bbest move\b|\bbest\b|\boptimal\b|\bperfect\b", MoveClassification.BEST),
    ]

    for pattern, classification in patterns:
        if re.search(pattern, text_lower):
            return classification

    # Check for sentiment
    negative_words = ["bad", "poor", "weak", "terrible", "awful", "wrong"]
    positive_words = ["good", "nice", "solid", "fine", "reasonable", "decent"]

    negative_count = sum(1 for word in negative_words if word in text_lower)
    positive_count = sum(1 for word in positive_words if word in text_lower)

    if negative_count > positive_count:
        return MoveClassification.INACCURACY  # Conservative default for negative
    elif positive_count > negative_count:
        return MoveClassification.GOOD

    return MoveClassification.UNKNOWN


def is_classification_compatible(
    llm_class: MoveClassification,
    engine_class: MoveClassification,
    tolerance: int = 1,
) -> bool:
    """Check if LLM classification is compatible with engine classification.

    Args:
        llm_class: Classification from LLM
        engine_class: Classification from engine
        tolerance: How many levels of difference is acceptable

    Returns:
        True if classifications are compatible
    """
    if llm_class == MoveClassification.UNKNOWN:
        return True  # Can't verify if unknown

    # Define classification order
    order = [
        MoveClassification.BEST,
        MoveClassification.GREAT,
        MoveClassification.GOOD,
        MoveClassification.INACCURACY,
        MoveClassification.MISTAKE,
        MoveClassification.BLUNDER,
    ]

    try:
        llm_idx = order.index(llm_class)
        engine_idx = order.index(engine_class)
        return abs(llm_idx - engine_idx) <= tolerance
    except ValueError:
        return True  # Unknown classification


async def verify_llm_response(
    fen: str,
    move: str,
    llm_response: str,
    stockfish: Stockfish,
    depth: int = 20,
) -> VerificationResult:
    """Verify an LLM response against engine analysis.

    Args:
        fen: Position in FEN notation
        move: Move being analyzed
        llm_response: LLM's response text
        stockfish: Stockfish instance
        depth: Analysis depth

    Returns:
        VerificationResult with validation details
    """
    # Get engine evaluation
    comparison = await stockfish.compare_moves(fen, move, depth)

    if "error" in comparison:
        return VerificationResult(
            valid=False,
            llm_classification=MoveClassification.UNKNOWN,
            engine_classification=MoveClassification.UNKNOWN,
            centipawn_loss=0,
            reason=f"Engine error: {comparison['error']}",
        )

    cp_loss = comparison.get("cp_loss", 0)
    engine_class = classify_move_by_cp_loss(cp_loss)
    llm_class = extract_classification_from_text(llm_response)

    # Check compatibility
    compatible = is_classification_compatible(llm_class, engine_class)

    if compatible:
        return VerificationResult(
            valid=True,
            llm_classification=llm_class,
            engine_classification=engine_class,
            centipawn_loss=cp_loss,
        )

    # Generate correction suggestion
    correction = _generate_correction(llm_class, engine_class, cp_loss, comparison)

    return VerificationResult(
        valid=False,
        llm_classification=llm_class,
        engine_classification=engine_class,
        centipawn_loss=cp_loss,
        reason=f"LLM said '{llm_class.value}' but engine says '{engine_class.value}'",
        correction=correction,
    )


def _generate_correction(
    llm_class: MoveClassification,
    engine_class: MoveClassification,
    cp_loss: int,
    comparison: dict,
) -> str:
    """Generate a correction message for incorrect LLM classification.

    Args:
        llm_class: LLM's classification
        engine_class: Engine's classification
        cp_loss: Centipawn loss
        comparison: Full comparison data

    Returns:
        Correction message
    """
    best_move = comparison.get("best_move", "")
    pv = comparison.get("pv", [])

    if engine_class in [MoveClassification.BLUNDER, MoveClassification.MISTAKE]:
        return (
            f"This move loses significant material or position. "
            f"The engine recommends {best_move} instead. "
            f"Key line: {' '.join(pv[:5])}"
        )
    elif engine_class == MoveClassification.INACCURACY:
        return (
            f"This move is slightly inaccurate. "
            f"A better option would be {best_move}."
        )
    elif engine_class in [MoveClassification.BEST, MoveClassification.GREAT]:
        return (
            f"This is actually a strong move! "
            f"It's close to the engine's top choice."
        )
    else:
        return f"The engine classifies this as '{engine_class.value}'."


class TacticalVerifier:
    """Verifier that checks LLM outputs against engine analysis."""

    def __init__(
        self,
        stockfish: Stockfish,
        regenerate_callback: Callable[[str, str, str], Awaitable[str]] | None = None,
        max_retries: int = 2,
    ):
        """Initialize the verifier.

        Args:
            stockfish: Stockfish instance
            regenerate_callback: Async function to regenerate LLM response
                                 Takes (fen, move, correction_hint) -> new_response
            max_retries: Maximum regeneration attempts
        """
        self.stockfish = stockfish
        self.regenerate_callback = regenerate_callback
        self.max_retries = max_retries

    async def verify_and_correct(
        self,
        fen: str,
        move: str,
        llm_response: str,
        depth: int = 20,
    ) -> tuple[str, VerificationResult]:
        """Verify response and regenerate if needed.

        Args:
            fen: Position in FEN notation
            move: Move being analyzed
            llm_response: Initial LLM response
            depth: Analysis depth

        Returns:
            Tuple of (final_response, verification_result)
        """
        current_response = llm_response

        for attempt in range(self.max_retries + 1):
            result = await verify_llm_response(
                fen, move, current_response, self.stockfish, depth
            )

            if result.valid:
                return current_response, result

            # Try to regenerate if callback provided
            if self.regenerate_callback and attempt < self.max_retries:
                hint = result.correction or f"Engine says: {result.engine_classification.value}"
                current_response = await self.regenerate_callback(fen, move, hint)
            else:
                # Return original with verification failure
                return current_response, result

        return current_response, result

    async def get_engine_context(self, fen: str, move: str, depth: int = 20) -> dict:
        """Get engine context that can be injected into LLM prompts.

        This provides the LLM with engine data to help it generate
        accurate responses without revealing raw scores.

        Args:
            fen: Position in FEN notation
            move: Move to analyze
            depth: Analysis depth

        Returns:
            Context dictionary for LLM prompting
        """
        comparison = await self.stockfish.compare_moves(fen, move, depth)

        if "error" in comparison:
            return {"error": comparison["error"]}

        cp_loss = comparison.get("cp_loss", 0)
        engine_class = classify_move_by_cp_loss(cp_loss)
        is_best = comparison.get("is_best", False)
        best_move = comparison.get("best_move", "")

        # Convert to natural language hints (without revealing scores)
        hints = []

        if is_best:
            hints.append("This is the engine's top choice.")
        elif engine_class == MoveClassification.GREAT:
            hints.append("This is a very strong move.")
        elif engine_class == MoveClassification.GOOD:
            hints.append("This is a reasonable move.")
        elif engine_class == MoveClassification.INACCURACY:
            hints.append("This move is slightly imprecise.")
            hints.append(f"Consider {best_move} as an alternative.")
        elif engine_class == MoveClassification.MISTAKE:
            hints.append("This move has some problems.")
            hints.append(f"The engine prefers {best_move}.")
        elif engine_class == MoveClassification.BLUNDER:
            hints.append("This move is a significant error.")
            hints.append(f"The correct move is {best_move}.")

        return {
            "classification": engine_class.value,
            "is_best": is_best,
            "best_move": best_move,
            "hints": hints,
            "refutation_line": comparison.get("pv", [])[:5],
        }
