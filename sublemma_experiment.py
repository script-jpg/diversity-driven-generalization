# %% Setup folder output path
# Comment out this part if not using Google drive
from google.colab import drive
drive.mount('/content/drive')

import os

# Define the path to the new folder in Google Drive
folder_path = '/content/drive/MyDrive/proofs_miniF2F' # set this to where you want the proofs saved

# Create the folder if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created successfully.")
else:
    print(f"Folder '{folder_path}' already exists.")
# %% Setup Lean environment with Mathlib

# Step 1: Install elan (Lean toolchain manager)
!curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y

# Step 2: Update Python process PATH so subprocess.run() can find `lean`
import os
elan_bin_path = os.path.expanduser("~/.elan/bin")
os.environ["PATH"] = elan_bin_path + ":" + os.environ["PATH"]

# Verify the installation by checking the version
!lean --version

import os
import subprocess

def setup_lean_project(project_dir="/tmp/lean_project"):
    """
    Creates a Lean project, configures it to use Mathlib,
    and downloads pre-compiled library files.
    """
    print(f"--- Setting up Lean project in: {project_dir} ---")
    os.makedirs(project_dir, exist_ok=True)

    # Content for the lakefile.lean
    lakefile_content = """
    import Lake
    open Lake DSL

    package «lean_project»

    require mathlib from git
      "https://github.com/leanprover-community/mathlib4.git"

    @[default_target]
    lean_lib «lean_project»
    """
    # Write the lakefile
    with open(os.path.join(project_dir, "lakefile.lean"), "w") as f:
        f.write(lakefile_content)

    # Run `lake exe cache get` to download Mathlib's pre-compiled files
    # This is much faster than building from source.
    print("--- Downloading Mathlib cache (this may take a few minutes)... ---")
    try:
        subprocess.run(
            ["lake", "exe", "cache", "get"],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True
        )
        print("--- Mathlib cache downloaded successfully. ---")
    except subprocess.CalledProcessError as e:
        print("❌ Error setting up Mathlib cache.")
        print(f"--- STDOUT ---\n{e.stdout}")
        print(f"--- STDERR ---\n{e.stderr}")
        raise  # Stop execution if setup fails

    return project_dir

# --- Call this function once at the start of your script ---
lean_project_path = setup_lean_project()
lean_project_path

# %% Import utility functions
import utils
print(utils.get_proof_variants)

# %% Function to check Lean proofs
import subprocess
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import get_proof_variants
# from typing import Dict
import threading

LOG_PATH = os.path.expanduser("~/error.log")   # expand ~ -> /home/you/...
os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
_log_lock = threading.Lock()

def check_lean_proof(proof_and_context: dict, log_errors=True) -> bool:
    """
    Checks a Lean‑4 proof string inside the given project using `lake`.
    If any variant succeeds, the *first* successful proof is saved to:
        corrected_proofs/<problem_id>/<proof_solver>/<attempt_id>.txt
    Returns True if a proof was saved, otherwise False.
    """
    # Verify the top‑level keys that must be present
    assert "proof" in proof_and_context, \
        "Missing 'proof' key – you need a proof string to test."
    assert "formal_statement" in proof_and_context, \
        "Missing 'formal_statement' key – you have to give the theorem statement."
    assert "project_dir" in proof_and_context, \
        "Missing 'project_dir' key – cannot locate the Lean project."
    assert "metadata" in proof_and_context, \
        "Missing 'metadata' key – you’ll need context such as attempt_id."

    # Verify the required nested keys inside metadata
    assert "attempt_id" in proof_and_context["metadata"], \
        "Metadata lacks 'attempt_id' – needed to name the output file."
    assert "problem_id" in proof_and_context["metadata"], \
        "Metadata lacks 'problem_id' – needed for the directory structure."
    assert "proof_solver" in proof_and_context["metadata"], \
        "Metadata lacks 'proof_solver' – you need to know which solver produced this."

    # Unpack everything we need
    proof_string   = proof_and_context["proof"]
    statement      = proof_and_context["formal_statement"]
    project_dir    = proof_and_context["project_dir"]

    metadata       = proof_and_context["metadata"]
    attempt_id     = metadata["attempt_id"]
    problem_id     = metadata["problem_id"]
    solver_name    = metadata["proof_solver"]


    # Where the successful proof will be written.
    save_dir = os.path.join(
        "corrected_proofs", problem_id, solver_name
    )
    os.makedirs(save_dir, exist_ok=True)          # make sure it exists
    

    # Build every candidate proof.
    proof_variants = get_proof_variants(proof_string)

    # Each variant becomes a tiny Lean file: statement + proof.
    candidates = [
        f"{statement}\n{variant}" for variant in proof_variants
    ]

    # Try them one by one.
    for idx, code in enumerate(candidates):
        temp_filename = f"{problem_id}_{solver_name}_{attempt_id}_{idx}.lean"
        temp_path = os.path.join(project_dir, temp_filename)

        try:
            # Write the candidate to a temporary file inside the project.
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(code)

            # Run Lean via lake.
            desired = 100_000
            command = [
                "lake", "env", "lean",
                f"-DmaxRecDepth={desired}",
                temp_filename
            ]
            result = subprocess.run(
                command,
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=120,               # 2 minutes, just in case
            )

            # Success = returncode 0 and no “error:” in stdout.
            if result.returncode == 0 and "error:" not in result.stdout:
                # Save the *first* working proof.
                out_path = os.path.join(save_dir, f"{attempt_id}.txt")
                with open(out_path, "w", encoding="utf-8") as out_f:
                    out_f.write(proof_variants[idx])

                # Clean up the temp file.
                os.remove(temp_path)

                return True   # yay, we found a good one
            if log_errors:
                if "error:" in result.stdout:
                    # print(attempt_id, result.stdout)
                    # thread-safe append
                    with _log_lock:
                        with open(LOG_PATH, "a", encoding="utf-8") as g:
                            g.writelines(result.stdout)

            # If it failed, just treat this variant as “false” and move on.
        except Exception as e:   # any crash = false for this variant
            # minimal logging: type and message, plus any subprocess output we have
            print(f"Exception ({type(e).__name__}): {e}")
            proc = locals().get("result")
            if proc is not None:
                print("---- subprocess stdout ----")
                print(proc.stdout or "<no stdout>")
                print("---- subprocess stderr ----")
                print(proc.stderr or "<no stderr>")
            # continue to next candidate
        finally:
            # Make sure we don’t leave stray temp files lying around.
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    # No variant succeeded.
    return False

# Example of using check_lean_proof
correct_proof_dict = {
    'formal_statement': 'import Mathlib.Tactic\ntheorem two_plus_two_is_four : 2 + 2 = 4',
    'proof': ':= by rfl',
    'project_dir': lean_project_path,
    'metadata': {'proof_solver': 'example_solver', 'problem_id': 'example_id', 'attempt_id': '1'}
}

check_lean_proof(correct_proof_dict)
# %% Lean helper functions
def split_formal_statement(formal_statement: str) -> tuple[str, str]:
    """
    Splits a formal statement into header and lemma parts.
    
    Args:
        formal_statement: A string containing import statements, opens, and a lemma/theorem
        
    Returns:
        A tuple of (header, lemma) where:
        - header contains all import and open statements
        - lemma contains the lemma/theorem declaration and its signature
    """
    lines = formal_statement.strip().split('\n')
    
    # Find the first line that starts with 'lemma', 'theorem', 'def', or 'example'
    lemma_start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(('lemma ')):
            lemma_start_idx = i
            break
    
    # Split into header and lemma
    header_lines = lines[:lemma_start_idx]
    lemma_lines = lines[lemma_start_idx:]
    
    # Join back into strings
    header = '\n'.join(header_lines).strip()
    lemma = '\n'.join(lemma_lines).strip()
    
    return header, lemma

# Test the function
formal_statement = """import Mathlib

open Real Nat Topology Complex
open scoped BigOperators

lemma h_cos_add (m n : ℝ) (k : ℕ) (a : ℕ → ℝ) (y : ℝ → ℝ) (h0 : 0 < k)
(h1 : ∀ x, y x = ∑ i ∈ Finset.range k, (Real.cos (a i + x)) / (2^i))
(h2 : y m = 0) (h3 : y n = 0) : ∀ i x, Real.cos (a i + x) = Real.cos (a i) * Real.cos x - Real.sin (a i) * Real.sin x := by"""

header, lemma = split_formal_statement(formal_statement)
print("Header:")
print(header)
print("\nLemma:")
print(lemma)

# %% Function to check Lean proofs and update proof cache
def check_lean_proof_and_update_proof_cache(proof_cache: list, proof_and_context: dict, log_errors=True) -> bool:
    if check_lean_proof(proof_and_context, log_errors):
        proved_lemma = split_formal_statement(proof_and_context['formal_statement'])[1] + " sorry" # proved by `sorry`
        proof_cache.append(proved_lemma)
        return True
    return False

# %% Functions to generate proofs
from transformers import pipeline, AutoConfig, AutoTokenizer, AutoModelForCausalLM
import logging

def _load_model(model_id):
    """
    Loads a single model and tokenizer to the GPU, with a fix for rope_scaling issues.
    """
    print(f"Attempting to load model: {model_id}")
    try:
        # 1. Load configuration first
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

        # 3. Load model and tokenizer with the (potentially corrected) config
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config, # Pass the corrected config
            torch_dtype="auto",
            trust_remote_code=True
        ).to("cuda")

        print(f"Successfully loaded {model_id}")
        return model, tok

    except Exception as e:
        # Provide a more informative error message
        logging.error(f"❌ Failed to load model '{model_id}'. Error: {e}")
        # Return None to be handled by the calling function, preventing the crash
        return None, None


def generate_prompt(proof_cache: list[str], formal_statement: str) -> str:
    informal_prefix = "-- Below are some previously proved lemmas that might help:\n"
    for lemma in proof_cache:
        informal_prefix += f"{lemma}\n"
    informal_prefix += "-- Now, using the above lemmas if needed, provide a proof for the following statement:\n"
    prompt = f"{informal_prefix}{formal_statement}\n"
    return prompt

def generate_proof(proof_cache, pipe, header, informal_prefix, formal_statement,
                   temperature: float = 0.5, max_new_tokens: int = 4096,
                   num_return_sequences: int = 1):
    
    prompt = generate_prompt(proof_cache, formal_statement)

    # Prepare arguments for the pipeline
    generation_args = {
        'do_sample': True,
        'eos_token_id': pipe.tokenizer.eos_token_id,
        'num_return_sequences': num_return_sequences,
        
    }

    # Only add max_new_tokens if a value is provided
    # If it remains None, the pipeline will use its own default
    if max_new_tokens is not None:
        generation_args['max_new_tokens'] = max_new_tokens

    # Call the pipeline with the arguments
    out = pipe(prompt, **generation_args)

    proofs = [result['generated_text'][len(prompt):].strip() for result in out]
    return proofs

from pathlib import Path
import gc
import torch
from tqdm import tqdm
from transformers import pipeline

def _sanitize_dir_name(name: str) -> str:
    return str(name).replace("/", "_").replace("\\", "_")

def _next_index(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    nums = []
    for p in out_dir.glob("*.txt"):
        stem = p.stem
        if stem.isdigit():
            nums.append(int(stem))
    return (max(nums) + 1) if nums else 1

def generate_proofs_memory_safe(
    proof_cache,
    model_ids,
    problem_row,
    problem_key,                 # e.g., DataFrame index or a unique ID column
    max_attempts: int,
    base_output_dir: str, # Added base_output_dir
    gpu_batch_size: int = 8,
    clear = True
):
    """
    Generate proofs and write to base_output_dir/<problem_key>/<model_id>/1.txt, 2.txt, ...
    No proof checking; purely generation + IO. Memory-safe (loads one model at a time).
    """
    for model_id in tqdm(model_ids, desc="Models"):
        model = tok = pipe = None
        attempt_bar = None
        try:
            model, tok = _load_model(model_id)
            pipe = pipeline("text-generation", model=model, tokenizer=tok, device=0)

            attempts_left = max_attempts
            attempt_bar = tqdm(total=max_attempts, desc=f"Generating {model_id}", leave=False)

            # Prepare output directory and next index (continues numbering if rerun)
            out_dir = Path(base_output_dir) / _sanitize_dir_name(problem_key) / _sanitize_dir_name(model_id) # Modified out_dir
            next_idx = _next_index(out_dir)

            print(next_idx)

            # raise Exception("Stop here")

            while attempts_left > 0:
                current_batch_size = min(gpu_batch_size, attempts_left)
                with torch.no_grad():
                    proof_snippets = generate_proof(
                        proof_cache,
                        pipe,
                        formal_statement=problem_row['formal_statement'],
                        num_return_sequences=current_batch_size
                    )

                # Write each snippet to numbered files 1.txt, 2.txt, ...
                for snippet in proof_snippets:
                    out_path = out_dir / f"{next_idx}.txt"
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(snippet)
                    next_idx += 1

                attempts_left -= current_batch_size
                attempt_bar.update(current_batch_size)

        finally:
          if clear:
            if attempt_bar is not None:
                attempt_bar.close()
            if model: del model
            if tok:   del tok
            if pipe:  del pipe
            gc.collect()
            torch.cuda.empty_cache()

def write_proofs_for_model(
    proof_cache,
    model_id: str,
    dataframe,
    base_output_dir: str, # Added base_output_dir
    max_attempts: int = 8,
    gpu_batch_size: int = 8,
    clear = True
):
    """
    For each problem in `dataframe`, generate `max_attempts` proofs for `model_id`
    and write them to base_output_dir/<problem_key>/<model_id>/*.txt.
    `problem_key` defaults to the DataFrame index value.
    """
    print(f"--- Generating proofs for model: {model_id} ---")
    for idx, problem_row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"Problems for {model_id}"):
        problem_key = problem_row.get('problem_id', idx)  # prefer a column named 'problem_id' if present
        generate_proofs_memory_safe(
            proof_cache,
            model_ids=[model_id],
            problem_row=problem_row,
            problem_key=problem_key,
            max_attempts=max_attempts,
            base_output_dir=base_output_dir, # Passed base_output_dir
            gpu_batch_size=gpu_batch_size,
            clear = clear
        )
    print(f"--- Done for {model_id}. ---")

# %% Single Model Eval Loop
proof_cache = []
solver_model_ids = [
    "Goedel-LM/Goedel-Prover-SFT",
    "AI-MO/Kimina-Prover-Preview-Distill-7B",
    "deepseek-ai/DeepSeek-Prover-V2-7B",
    "deepseek-ai/DeepSeek-Prover-V1.5-RL"
]

from datasets import load_dataset
folder_path = ""
# from google.colab import files
# import time

miniF2F_test_df = load_dataset("script-jpg/imo-1969-p2-lemmas", split="train").to_pandas()

for i, mid in enumerate(solver_model_ids):
    try:
      write_proofs_for_model(proof_cache, mid, miniF2F_test_df, base_output_dir=folder_path, max_attempts=8, gpu_batch_size=4, clear=True) # Passed folder_path
      
    except Exception as e:
      print(f"Error for {mid}: {e}")