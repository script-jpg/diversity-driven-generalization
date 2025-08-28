#!/usr/bin/env python3
"""
convert_proofs.py

Bidirectional conversion between

    1. proof_outcomes_by_problem.json   # {problem_id -> {model_id -> [8 ints]}}
    2. proof_outcomes_by_model.json    # {model_id   -> {problem_id -> [8 ints]}}

Usage
-----

    python convert_proofs.py to-model   <in_json>  <out_json>
    python convert_proofs.py to-problem <in_json>  <out_json>
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any


# ----------------------------------------------------------------------
# Helper types for readability
# ----------------------------------------------------------------------
ProblemID = str           # e.g. "0", "1", … "39"
ModelID   = str           # e.g. "AI-MO_Kimina-Prover-Preview-Distill-7B"
Attempt   = List[int]     # length‑8 list of 0/1


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file, raise a nice error on failure."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"❗ Failed to read JSON from {path}: {exc}\n")
        sys.exit(1)


def dump_json(data: Dict[str, Any], path: Path) -> None:
    """Write JSON with 2‑space indent, preserve ordering of keys."""
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"❗ Could not write JSON to {path}: {exc}\n")
        sys.exit(1)


# ----------------------------------------------------------------------
# Conversion functions
# ----------------------------------------------------------------------
def to_model(problem_json: Dict[ProblemID, Dict[ModelID, Attempt]]) -> Dict[ModelID, Dict[ProblemID, Attempt]]:
    """
    Transform {problem → model → attempts} into {model → problem → attempts}.
    """
    model_dict: Dict[ModelID, Dict[ProblemID, Attempt]] = {}

    for prob_id, model_map in problem_json.items():
        for model_id, attempts in model_map.items():
            # Ensure the inner list is exactly length‑8 (defensive)
            if not isinstance(attempts, list) or len(attempts) != 8:
                sys.stderr.write(
                    f"⚠️  Warning: attempts for problem {prob_id}, model {model_id} "
                    f"are not a length‑8 list. They will be padded/truncated.\n"
                )
                attempts = (attempts[:8] + [0] * 8)[:8]   # pad/truncate

            model_dict.setdefault(model_id, {})[prob_id] = attempts

    # Sort keys for reproducibility (optional, but matches the example layout)
    return {
        model_id: dict(sorted(prob_map.items(), key=lambda kv: int(kv[0])))
        for model_id, prob_map in sorted(model_dict.items())
    }


def to_problem(model_json: Dict[ModelID, Dict[ProblemID, Attempt]]) -> Dict[ProblemID, Dict[ModelID, Attempt]]:
    """
    Transform {model → problem → attempts} back into {problem → model → attempts}.
    """
    problem_dict: Dict[ProblemID, Dict[ModelID, Attempt]] = {}

    for model_id, prob_map in model_json.items():
        for prob_id, attempts in prob_map.items():
            if not isinstance(attempts, list) or len(attempts) != 8:
                sys.stderr.write(
                    f"⚠️  Warning: attempts for model {model_id}, problem {prob_id} "
                    f"are not a length‑8 list. They will be padded/truncated.\n"
                )
                attempts = (attempts[:8] + [0] * 8)[:8]

            problem_dict.setdefault(prob_id, {})[model_id] = attempts

    # Sort both levels numerically / alphabetically for a deterministic output
    return {
        prob_id: dict(sorted(model_map.items()))
        for prob_id, model_map in sorted(problem_dict.items(), key=lambda kv: int(kv[0]))
    }


# ----------------------------------------------------------------------
# CLI handling
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert between problem‑centric and model‑centric proof outcome JSON files."
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # to-model
    p_to_model = sub.add_parser("to-model", help="Convert problem→model JSON to model→problem JSON")
    p_to_model.add_argument("in_path", type=Path, help="Path to the problem‑centric JSON")
    p_to_model.add_argument("out_path", type=Path, help="Destination for the model‑centric JSON")

    # to-problem
    p_to_problem = sub.add_parser(
        "to-problem", help="Convert model→problem JSON back to problem→model JSON"
    )
    p_to_problem.add_argument("in_path", type=Path, help="Path to the model‑centric JSON")
    p_to_problem.add_argument("out_path", type=Path, help="Destination for the problem‑centric JSON")

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "to-model":
        src = load_json(args.in_path)
        out = to_model(src)                  # type: ignore[arg-type]
        dump_json(out, args.out_path)
        print(f"✅  Saved model‑centric JSON to {args.out_path}")

    elif args.command == "to-problem":
        src = load_json(args.in_path)
        out = to_problem(src)                # type: ignore[arg-type]
        dump_json(out, args.out_path)
        print(f"✅  Saved problem‑centric JSON to {args.out_path}")

    else:                                 # pragma: no cover
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
