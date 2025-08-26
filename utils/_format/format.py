from __future__ import annotations
import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Union, TypeVar, Generic
import sys

logger = logging.getLogger(__name__)

# --- Result Type ---
T = TypeVar("T")
E = TypeVar("E")


@dataclass(frozen=True)
class Ok(Generic[T]):
    value: T


@dataclass(frozen=True)
class Err(Generic[E]):
    error: E


Result = Union[Ok[T], Err[E]]


# --- Helpers ---
def strip_trailing_fence(text: str) -> str:
    lines = text.splitlines()
    idx = len(lines) - 1
    while idx >= 0 and not lines[idx].strip():
        idx -= 1
    if idx >= 0 and lines[idx].strip().startswith("```"):
        return "\n".join(lines[:idx]).rstrip()
    return text


def extract_last_lean4_block(text: str) -> Optional[str]:
    pattern = re.compile(
        r"""^[ \t]*```          # opening fence
            (?:lean4?|lean)    # language tag
            [ \t]*\r?\n        # newline after tag
            (.*?)              # code (lazy)
            ^[ \t]*```[ \t]*$  # closing fence
        """,
        re.DOTALL | re.MULTILINE | re.IGNORECASE | re.VERBOSE,
    )
    matches = pattern.findall(text)
    return matches[-1].strip() if matches else None


def remove_leading_whitespace(s: str, count: int | None = None) -> str:
    lines = s.split("\n")
    if len(lines) <= 1:
        return s
    if count is None:
        # count spaces on the second line only
        count = len(lines[1]) - len(lines[1].lstrip(" "))
    return "\n".join([lines[0]] + [line[count:] for line in lines[1:]])


def remove_trailing_end(s: str) -> str:
    lines = s.split("\n")
    if lines and lines[-1].strip() == "end":
        lines.pop()
    return "\n".join(lines)


def format_proof(s: str, count: int | None = None) -> str:
    return remove_leading_whitespace(
        remove_trailing_end(strip_trailing_fence(s)),
        count,
    )


# ---------- Safe wrapper ----------
def safe_format_proof(s: str, count: int | None = None) -> Result[str, Exception]:
    try:
        return Ok(format_proof(s, count))
    except Exception as exc:  # pylint: disable=broad-except
        return Err(exc)


# ---------- Bulk strategy using the Result wrapper ----------
def apply_bulk_strategies(s: str) -> List[str]:
    match extract_last_lean4_block(s):
        case None:
            # Try (variable, 2, 4) indent
            # Each call is wrapped in `safe_format_proof` which yields a Result[Ok|Err].
            attempts: List[Result[str, Exception]] = (
                [safe_format_proof(s)] +
                [safe_format_proof(s, i) for i in (2, 4)]
            )
        case block:
            # We found a Lean block – format it once.
            attempts: List[Result[str, Exception]] = [safe_format_proof(block)]

    # Keep only the successful proofs; optionally log the failures.
    results: List[str] = []
    for r in attempts:
        if isinstance(r, Ok):
            results.append(r.value)
        else:
            logger.debug("formatProof failed with %s", r.error)

    return results


def get_proof_variants(s: str) -> List[str]:
    return [s] + apply_bulk_strategies(s)


if __name__ == "__main__":
    # ---
    # Command‑line handling
    # ---
    if len(sys.argv) < 2:
        print("Usage: python script.py <file>", file=sys.stderr)
        sys.exit(1)

    filename = sys.argv[1]

    try:
        with open(filename, "r", encoding="utf-8", errors="replace") as f:
            pf = f.read()
            print("\n---\n".join(get_proof_variants(pf)))
    except OSError as exc:
        print(f"❌ Unable to read {filename!r}: {exc}", file=sys.stderr)
        sys.exit(2)
