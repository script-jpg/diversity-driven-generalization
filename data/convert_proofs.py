#!/usr/bin/env python3
"""
convert_proofs.py

Four-way conversion:

    1. problem → model         (to-model)
    2. model   → problem      (to-problem)
    3. model   → checkpoint   (to-checkpoint)
    4. checkpoint → model     (from-checkpoint)
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any

# ----------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------
ProblemID = str            # "0" … "39"
ModelID   = str            # e.g. "AI-MO_Kimina-Prover-Preview-Distill-7B"
Attempt   = List[int]      # length‑8 list of 0/1


# ----------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------
def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"❗ Failed to read JSON from {path}: {exc}\n")
        sys.exit(1)


def dump_json(data: Dict[str, Any], path: Path) -> None:
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"❗ Could not write JSON to {path}: {exc}\n")
        sys.exit(1)


# ----------------------------------------------------------------------
# Core conversion functions
# ----------------------------------------------------------------------
def to_model(problem_json: Dict[ProblemID, Dict[ModelID, Attempt]]) -> Dict[ModelID, Dict[ProblemID, Attempt]]:
    """{problem → model → attempts}  →  {model → problem → attempts}."""
    model_dict: Dict[ModelID, Dict[ProblemID, Attempt]] = {}
    for prob_id, model_map in problem_json.items():
        for model_id, attempts in model_map.items():
            attempts = (attempts[:8] + [0] * 8)[:8]  # defensive pad/truncate
            model_dict.setdefault(model_id, {})[prob_id] = attempts
    # deterministic ordering
    return {
        model_id: dict(sorted(prob_map.items(), key=lambda kv: int(kv[0])))
        for model_id, prob_map in sorted(model_dict.items())
    }


def to_problem(model_json: Dict[ModelID, Dict[ProblemID, Attempt]]) -> Dict[ProblemID, Dict[ModelID, Attempt]]:
    """{model → problem → attempts}  →  {problem → model → attempts}."""
    prob_dict: Dict[ProblemID, Dict[ModelID, Attempt]] = {}
    for model_id, prob_map in model_json.items():
        for prob_id, attempts in prob_map.items():
            attempts = (attempts[:8] + [0] * 8)[:8]
            prob_dict.setdefault(prob_id, {})[model_id] = attempts
    return {
        prob_id: dict(sorted(model_map.items()))
        for prob_id, model_map in sorted(prob_dict.items(), key=lambda kv: int(kv[0]))
    }


def model_to_checkpoint(model_json: Dict[ModelID, Dict[ProblemID, Attempt]]) -> Dict[str, Dict[ModelID, str]]:
    """
    Build the tiny checkpoint format.

    For each model we emit a binary string of length = number of problems.
    Position *i* is '1' iff any of the eight attempts for problem *i* is 1.
    """
    # Determine the set of problem IDs and sort them numerically.
    all_probs = sorted(
        {int(p) for prob_map in model_json.values() for p in prob_map.keys()}
    )
    # sanity: we expect 40 problems, but we don’t enforce it.
    model_dict: Dict[ModelID, str] = {}

    for model_id, prob_map in model_json.items():
        bits: List[str] = []
        for prob_id in map(str, all_probs):
            attempts = prob_map.get(prob_id, [0] * 8)
            bits.append("1" if any(attempts) else "0")
        model_dict[model_id] = "".join(bits)

    return {"model_dict": model_dict}


def checkpoint_to_model(chk_json: Dict[str, Dict[ModelID, str]]) -> Dict[ModelID, Dict[ProblemID, Attempt]]:
    """
    Reverse a checkpoint back to the model‑centric format.

    Because the checkpoint loses the per‑attempt granularity we fabricate a
    placeholder attempts list:
        bit == '1' → [1,0,0,0,0,0,0,0]
        bit == '0' → [0,0,0,0,0,0,0,0]

    The resulting JSON is a *best‑effort* reconstruction; it won’t match the
    original attempts byte‑for‑byte, but it satisfies the “at least one 1”
    invariant.
    """
    model_dict: Dict[ModelID, Dict[ProblemID, Attempt]] = {}

    for model_id, bitstr in chk_json.get("model_dict", {}).items():
        prob_map: Dict[ProblemID, Attempt] = {}
        for idx, ch in enumerate(bitstr):
            prob_id = str(idx)                     # problems are 0‑indexed
            prob_map[prob_id] = [1, 0, 0, 0, 0, 0, 0, 0] if ch == "1" else [0] * 8
        model_dict[model_id] = prob_map

    return model_dict


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert between problem‑centric, model‑centric and checkpoint proof JSON formats."
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # problem → model
    p_to_model = sub.add_parser("to-model", help="Convert problem‑centric JSON to model‑centric JSON")
    p_to_model.add_argument("in_path", type=Path, help="Path to problem‑centric JSON")
    p_to_model.add_argument("out_path", type=Path, help="Destination for model‑centric JSON")

    # model → problem
    p_to_problem = sub.add_parser("to-problem", help="Convert model‑centric JSON back to problem‑centric JSON")
    p_to_problem.add_argument("in_path", type=Path, help="Path to model‑centric JSON")
    p_to_problem.add_argument("out_path", type=Path, help="Destination for problem‑centric JSON")

    # model → checkpoint
    p_to_chk = sub.add_parser(
        "to-checkpoint",
        help="Create the tiny checkpoint.json from a model‑centric JSON file"
    )
    p_to_chk.add_argument("in_path", type=Path, help="Model‑centric JSON")
    p_to_chk.add_argument("out_path", type=Path, help="Where to write checkpoint.json")

    # checkpoint → model
    p_from_chk = sub.add_parser(
        "from-checkpoint",
        help="Re‑construct a model‑centric JSON from a checkpoint.json (best‑effort)"
    )
    p_from_chk.add_argument("in_path", type=Path, help="Checkpoint JSON")
    p_from_chk.add_argument("out_path", type=Path, help="Destination for model‑centric JSON")

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "to-model":
        src = load_json(args.in_path)
        out = to_model(src)                     # type: ignore[arg-type]
        dump_json(out, args.out_path)
        print(f"✅  Model‑centric JSON written to {args.out_path}")

    elif args.command == "to-problem":
        src = load_json(args.in_path)
        out = to_problem(src)                   # type: ignore[arg-type]
        dump_json(out, args.out_path)
        print(f"✅  Problem‑centric JSON written to {args.out_path}")

    elif args.command == "to-checkpoint":
        src = load_json(args.in_path)
        out = model_to_checkpoint(src)          # type: ignore[arg-type]
        dump_json(out, args.out_path)
        print(f"✅  Checkpoint JSON written to {args.out_path}")

    elif args.command == "from-checkpoint":
        src = load_json(args.in_path)
        out = checkpoint_to_model(src)          # type: ignore[arg-type]
        dump_json(out, args.out_path)
        print(f"✅  Re‑constructed model‑centric JSON written to {args.out_path}")

    else:  # pragma: no cover
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
