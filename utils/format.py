from typing import Optional
import sys
import re


def strip_trailing_fence(text: str) -> str:
    """
    If the last non-empty line is a markdown fence (like ``` or ```python),
    remove it and return the rest of the text.
    """
    lines = text.splitlines()

    # Work backwards to skip trailing empty lines
    idx = len(lines) - 1
    while idx >= 0 and lines[idx].strip() == "":
        idx -= 1

    if idx >= 0 and lines[idx].strip().startswith("```"):
        # Slice off the fence line and keep earlier lines (including blank ones before it)
        return "\n".join(lines[:idx]).rstrip()

    return text


def extract_last_lean4_block(text: str) -> Optional[str]:
    """
    Return the *inner* source of the **last** fenced code block whose language
    identifier is ``lean`` or ``lean4`` (case‑insensitive).

    The function tolerates optional whitespace before/after the fences and
    works with both LF (``\\n``) and CRLF (``\\r\\n``) line endings.

    If no such block exists, ``None`` is returned.
    """
    # Regex explanation (using the verbose flag for readability):
    #   ^[ \t]*```           – opening fence, possibly indented
    #   (?:lean4?|lean)      – language tag (lean, lean3, lean4, …)
    #   [ \t]*\r?\n          – optional spaces then a newline (start of code)
    #   (.*?)                – **capture group 1**: everything lazily up to the closing fence
    #   ^[ \t]*```[ \t]*$    – closing fence on its own line, optional spaces
    #
    # Flags:
    #   re.DOTALL  – makes '.' match newlines (so (.*?) spans multiple lines)
    #   re.MULTILINE – '^' and '$' work per line, not just at the whole‑string ends
    #   re.IGNORECASE – allow LEAN, Lean4, etc.
    pattern = re.compile(
        r"""^[ \t]*```          # opening fence (maybe indented)
            (?:lean4?|lean)    # language name
            [ \t]*\r?\n        # end of the opening line
            (.*?)              # *** capture the code ***
            ^[ \t]*```[ \t]*$  # closing fence on its own line
        """,
        re.DOTALL | re.MULTILINE | re.IGNORECASE | re.VERBOSE,
    )

    # `findall` returns a list of the captured group (the code itself)
    matches = pattern.findall(text)

    # Return the *last* match, stripped of surrounding blank lines
    return matches[-1].strip() if matches else None


def remove_leading_whitespace(s: str, count: int | None = None) -> str:
    """
    Finds leading whitespace on second line and then subtracts by that for all remaining lines
    """
    # assert (count is None or count >= 0, "count must be non-negative")
    lines = s.split("\n")
    if len(lines) <= 1:
        return s
    else:
        # get leading whitespace on second line
        p = 0
        if count is None:
            while lines[1][p] == " ":
                p += 1
        else:
            p = count

        # return first line unchanged along with the modified remaining lines
        return "\n".join([lines[0]] + [line[p:] for line in lines[1:]])


def remove_trailing_end(s: str) -> str:
    """
    This function splits the lines of s and then removes the last line if it is only 'end'
    """
    lines = s.split("\n")
    if lines and lines[-1].strip() == "end":
        lines.pop()
    return "\n".join(lines)


def formatProof(s: str, count: int | None = None) -> str:
    return remove_trailing_end(remove_leading_whitespace(strip_trailing_fence(s), count))


def apply_bulk_strategies(s: str) -> list[str]:
    match extract_last_lean4_block(s):
        case None:
            # no lean block inside
            # try variable + 2 + 4
            return [formatProof(s)] + [formatProof(s, i) for i in (2, 4)]
        case block:
            return [formatProof(block)]


def get_proof_variants(s: str) -> list[str]:
    return [s] + apply_bulk_strategies(s)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1️⃣  Command‑line handling
    # ------------------------------------------------------------------
    if len(sys.argv) < 2:
        print("Usage: python script.py <file>", file=sys.stderr)
        sys.exit(1)

    filename = sys.argv[1]

    # ------------------------------------------------------------------
    # 2️⃣  Read the file **with the correct encoding** (utf‑8 handles the
    #     subscript, emojis, etc.).  Use `errors='replace'` if you want to
    #     silently substitute any stray bytes.
    # ------------------------------------------------------------------
    try:
        with open(filename, "r", encoding="utf-8", errors="replace") as f:
            pf = f.read()
            print("\n---\n".join(get_proof_variants(pf)))
    except OSError as exc:
        print(f"❌ Unable to read {filename!r}: {exc}", file=sys.stderr)
        sys.exit(2)
