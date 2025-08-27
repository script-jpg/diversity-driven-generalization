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


def extract_proof_from_full(s: str) -> Result[str, Exception]:
    """
    sometimes the entire proof is inside the lean block. We need to extract just the proof.
    """
    try:
        m_thm = re.search(r"\btheorem\b", s)
        if not m_thm:
            return Err(Exception("no theorem keyword found"))
        m_by = re.search(r"\bby\b", s[m_thm.end() :])
        if not m_by:
            return Err(Exception("no `by` found after theorem"))
        start = m_thm.end() + m_by.start() + 2  # position right after 'by'
        proof = s[start:].lstrip("\n ")  # keep comments; just trim leading whitespace
        if not proof.strip():
            return Err(Exception("empty proof after `by`"))
        return Ok(proof)
    except Exception as exc:
        return Err(exc)


# ---------- Bulk strategy using the Result wrapper ----------
def apply_bulk_strategies(s: str) -> List[str]:
    pf: str = ""
    match extract_last_lean4_block(s):
        case None:
            pf = s
        case block:
            extracted_proof = extract_proof_from_full(block)
            match extracted_proof:
                case Ok(val):
                    pf = val
                case Err(_):
                    pf = block

    attempts: List[Result[str, Exception]] = (
        [Ok(pf)] + [safe_format_proof(pf)] + [safe_format_proof(pf, i) for i in (2, 4)]
    )

    # Keep only the successful proofs; optionally log the failures.
    results: List[str] = []
    for r in attempts:
        match r:
            case Ok(val):
                results.append(val)
            case Err(e):
                logger.debug("formatProof failed with %s", e)

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
