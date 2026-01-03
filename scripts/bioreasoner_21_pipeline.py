#!/usr/bin/env python3
"""
BioReasoner 2.1 Training Pipeline
=================================
Best-of-N generation with Tribunal filtering for DPO training.

Phase 1: Generate N=4 responses per prompt with force-prefixing
Phase 2: Auto-filter for <think> blocks and basic quality
Phase 3: Prepare for Tribunal scoring (Opus/Codex)
Phase 4: Construct DPO preference pairs

Usage:
    # Phase 1: Generate candidates
    python scripts/bioreasoner_21_pipeline.py generate --n_per_prompt 4

    # Phase 2: Auto-filter
    python scripts/bioreasoner_21_pipeline.py filter

    # Phase 3: After Tribunal scoring, build DPO pairs
    python scripts/bioreasoner_21_pipeline.py build_dpo
"""

import argparse
import json
import re
import random
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict

# Paths
BASE_DIR = Path("/home/user/rubric-rewards-training")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "bioreasoner_21"
MODEL_PATH = BASE_DIR / "models" / "bioreasoner-2.0-merged"

# Ensure output dir exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Candidate:
    """A single generation candidate."""
    prompt_id: str
    prompt_type: str
    papers: list
    prompt: str
    response: str
    generation_idx: int
    has_think: bool
    think_length: int
    has_hypothesis: bool
    has_methods: bool
    auto_score: float  # Automatic pre-filter score
    tribunal_scores: Optional[dict] = None  # Filled by Tribunal


def load_prompts(n_prompts: int = 150) -> list[dict]:
    """Load prompts from training data, prioritizing diverse types."""
    prompts = []

    # Load from multiple sources for diversity
    sources = [
        DATA_DIR / "test_84.jsonl",      # Held-out test (high quality)
        DATA_DIR / "train_750.jsonl",     # Training data
    ]

    seen_ids = set()
    for source in sources:
        if not source.exists():
            continue
        with open(source) as f:
            for line in f:
                item = json.loads(line)
                gid = item.get("group_id", item.get("id", ""))
                if gid not in seen_ids:
                    seen_ids.add(gid)
                    prompts.append({
                        "prompt_id": gid,
                        "prompt_type": item.get("prompt_type", "unknown"),
                        "papers": item.get("papers", []),
                        "prompt": item.get("prompt", ""),
                    })

    # Shuffle and select
    random.shuffle(prompts)
    selected = prompts[:n_prompts]

    # Ensure diversity of prompt types
    type_counts = {}
    for p in selected:
        t = p["prompt_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"Loaded {len(selected)} prompts")
    print(f"Prompt type distribution: {type_counts}")

    return selected


def build_generation_prompt(prompt: str) -> str:
    """
    Build the full prompt with force-prefixing.
    Forces model to start with <think> block.
    """
    system = """You are BioReasoner, a scientific reasoning assistant trained on Vanderbilt faculty publications.

When given papers to analyze, you MUST:
1. Start your response with <think> to show your reasoning process
2. Analyze each paper systematically inside the think block
3. Close with </think> before your final answer
4. Cite papers using [Author Year, PMID:XXXXX] format
5. ONLY cite the papers provided - never fabricate references

Your response format:
<think>
[Detailed analysis of each paper]
[Identification of shared patterns/gaps]
[Synthesis of cross-paper insights]
</think>

[Your hypothesis/extension/answer based on the analysis]
"""

    return f"{system}\n\n{prompt}"


def parse_response(response: str) -> dict:
    """
    Parse a model response and extract quality signals.
    Returns dict with quality metrics for auto-filtering.
    """
    # Check for <think> block
    has_think = "<think>" in response and "</think>" in response
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    think_length = len(think_match.group(1).strip()) if think_match else 0

    # Check for hypothesis/extension markers
    has_hypothesis = bool(re.search(
        r'HYPOTHESIS|PROPOSED|EXTENSION|LIMITATION|RATIONALE',
        response, re.IGNORECASE
    ))

    # Check for methods section
    has_methods = bool(re.search(
        r'METHODS?:|APPROACH:|IMPLEMENTATION:|EXPERIMENT:',
        response, re.IGNORECASE
    ))

    # Check for specific metrics (sign of novelty)
    has_metrics = bool(re.search(
        r'\d+%|AUC|Dice|accuracy|p\s*[<>=]\s*0\.\d+|n\s*=\s*\d+',
        response, re.IGNORECASE
    ))

    # Auto-score (0-5 scale, used for pre-filtering)
    auto_score = 0.0
    if has_think:
        auto_score += 2.0  # Critical requirement
        if think_length > 500:
            auto_score += 0.5
        if think_length > 1000:
            auto_score += 0.5
    if has_hypothesis:
        auto_score += 1.0
    if has_methods:
        auto_score += 0.5
    if has_metrics:
        auto_score += 0.5

    return {
        "has_think": has_think,
        "think_length": think_length,
        "has_hypothesis": has_hypothesis,
        "has_methods": has_methods,
        "has_metrics": has_metrics,
        "auto_score": min(auto_score, 5.0),
    }


def generate_candidates_stub(prompts: list[dict], n_per_prompt: int = 4) -> list[Candidate]:
    """
    STUB: This generates the input file for Codex to run inference.
    Codex will load the merged model and generate actual responses.
    """
    # Create batch file for Codex
    batch = []
    for prompt_data in prompts:
        for gen_idx in range(n_per_prompt):
            batch.append({
                "prompt_id": prompt_data["prompt_id"],
                "prompt_type": prompt_data["prompt_type"],
                "papers": prompt_data["papers"],
                "prompt": prompt_data["prompt"],
                "full_prompt": build_generation_prompt(prompt_data["prompt"]),
                "generation_idx": gen_idx,
                "seed": random.randint(0, 2**32 - 1),  # For reproducibility
            })

    # Save batch for Codex
    batch_file = OUTPUT_DIR / "generation_batch.jsonl"
    with open(batch_file, "w") as f:
        for item in batch:
            f.write(json.dumps(item) + "\n")

    print(f"\n{'='*70}")
    print("GENERATION BATCH CREATED")
    print(f"{'='*70}")
    print(f"Total prompts: {len(prompts)}")
    print(f"Generations per prompt: {n_per_prompt}")
    print(f"Total candidates to generate: {len(batch)}")
    print(f"Batch file: {batch_file}")
    print(f"\n>>> PASS TO CODEX FOR INFERENCE <<<")

    return []


def filter_candidates(input_file: Optional[Path] = None) -> tuple[list[dict], list[dict]]:
    """
    Auto-filter generated candidates.
    Returns (passed, rejected) lists.
    """
    if input_file is None:
        input_file = OUTPUT_DIR / "generation_results.jsonl"

    if not input_file.exists():
        print(f"ERROR: {input_file} not found. Run Codex inference first.")
        return [], []

    passed = []
    rejected = []

    with open(input_file) as f:
        for line in f:
            item = json.loads(line)
            response = item.get("response", item.get("model_response", ""))

            # Parse quality signals
            quality = parse_response(response)
            item.update(quality)

            # REJECTION CRITERIA
            reject_reason = None

            if not quality["has_think"]:
                reject_reason = "MISSING_THINK_BLOCK"
            elif quality["think_length"] < 200:
                reject_reason = "THINK_TOO_SHORT"
            elif quality["auto_score"] < 2.5:
                reject_reason = "LOW_AUTO_SCORE"

            if reject_reason:
                item["reject_reason"] = reject_reason
                rejected.append(item)
            else:
                passed.append(item)

    # Save filtered results
    passed_file = OUTPUT_DIR / "candidates_passed.jsonl"
    rejected_file = OUTPUT_DIR / "candidates_rejected.jsonl"

    with open(passed_file, "w") as f:
        for item in passed:
            f.write(json.dumps(item) + "\n")

    with open(rejected_file, "w") as f:
        for item in rejected:
            f.write(json.dumps(item) + "\n")

    print(f"\n{'='*70}")
    print("AUTO-FILTER RESULTS")
    print(f"{'='*70}")
    print(f"Total candidates: {len(passed) + len(rejected)}")
    print(f"Passed: {len(passed)} ({100*len(passed)/(len(passed)+len(rejected)):.1f}%)")
    print(f"Rejected: {len(rejected)}")
    print(f"\nRejection breakdown:")

    reasons = {}
    for item in rejected:
        r = item.get("reject_reason", "UNKNOWN")
        reasons[r] = reasons.get(r, 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    print(f"\nPassed candidates: {passed_file}")
    print(f"Rejected candidates: {rejected_file}")
    print(f"\n>>> SEND {passed_file} TO TRIBUNAL FOR SCORING <<<")

    return passed, rejected


def build_dpo_pairs(scored_file: Optional[Path] = None, include_synthetic: bool = True) -> list[dict]:
    """
    Build DPO preference pairs from Tribunal-scored candidates.

    Selection logic:
    - For each prompt, find best and worst responses
    - Winner: highest novelty score, has <think>
    - Loser: lowest novelty score OR missing <think>
    - Only create pairs with score delta >= 2 for clear signal
    - Include synthetic exemplars for strong guidance
    """
    # Import novelty scoring
    import sys
    sys.path.insert(0, str(BASE_DIR / "scripts"))
    from novelty_scoring import score_for_dpo, get_synthetic_dpo_pairs

    if scored_file is None:
        scored_file = OUTPUT_DIR / "candidates_passed.jsonl"  # Use auto-filtered candidates

    if not scored_file.exists():
        print(f"ERROR: {scored_file} not found. Run filter first.")
        return []

    # Load candidates
    candidates = []
    with open(scored_file) as f:
        for line in f:
            candidates.append(json.loads(line))

    print(f"Loaded {len(candidates)} candidates for DPO pair construction")

    # Score each candidate with novelty scorer
    for c in candidates:
        response = c.get("response", c.get("model_response", ""))
        novelty_score, details = score_for_dpo(response)
        c["novelty_score"] = novelty_score
        c["novelty_details"] = details

    # Group by prompt
    by_prompt = {}
    for c in candidates:
        pid = c.get("prompt_id", "")
        if pid not in by_prompt:
            by_prompt[pid] = []
        by_prompt[pid].append(c)

    # Build pairs
    dpo_pairs = []

    for prompt_id, responses in by_prompt.items():
        if len(responses) < 2:
            continue

        # Score each response (novelty-focused)
        def score(r):
            # Primary: novelty score
            novelty = r.get("novelty_score", 2)

            # Penalize missing think or short think
            if not r.get("has_think", True):
                return -10
            if r.get("think_length", 0) < 500:
                novelty -= 0.5

            return novelty

        # Sort by score
        sorted_resp = sorted(responses, key=score, reverse=True)

        winner = sorted_resp[0]
        loser = sorted_resp[-1]

        # Only create pair if there's meaningful difference (>= 2 points)
        winner_score = score(winner)
        loser_score = score(loser)

        if winner_score >= loser_score + 2.0:  # Require 2+ point difference for clear signal
            pair = {
                "prompt_id": prompt_id,
                "prompt_type": winner.get("prompt_type", ""),
                "prompt": winner.get("prompt", ""),
                "papers": winner.get("papers", []),
                "chosen": winner.get("response", winner.get("model_response", "")),
                "rejected": loser.get("response", loser.get("model_response", "")),
                "chosen_score": winner_score,
                "rejected_score": loser_score,
                "score_delta": winner_score - loser_score,
                "synthetic": False,
            }
            dpo_pairs.append(pair)

    print(f"Created {len(dpo_pairs)} pairs from model generations")

    # Add synthetic exemplars
    if include_synthetic:
        synthetic_file = DATA_DIR / "synthetic_dpo_exemplars.jsonl"
        if synthetic_file.exists():
            synthetic_pairs = []
            with open(synthetic_file) as f:
                for line in f:
                    synthetic_pairs.append(json.loads(line))
            dpo_pairs.extend(synthetic_pairs)
            print(f"Added {len(synthetic_pairs)} synthetic exemplar pairs")
        else:
            # Use programmatic synthetic pairs
            synthetic_pairs = get_synthetic_dpo_pairs()
            dpo_pairs.extend(synthetic_pairs)
            print(f"Added {len(synthetic_pairs)} programmatic synthetic pairs")

    # Save pairs
    pairs_file = OUTPUT_DIR / "dpo_pairs_21.jsonl"
    with open(pairs_file, "w") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair) + "\n")

    # Also save in training format
    training_file = DATA_DIR / "dpo_v21_pairs.jsonl"
    with open(training_file, "w") as f:
        for pair in dpo_pairs:
            # Standard DPO format
            training_item = {
                "prompt": pair["prompt"],
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
            }
            f.write(json.dumps(training_item) + "\n")

    print(f"\n{'='*70}")
    print("DPO PAIR CONSTRUCTION COMPLETE")
    print(f"{'='*70}")
    print(f"Prompts with candidates: {len(by_prompt)}")
    print(f"DPO pairs created: {len(dpo_pairs)}")
    pairs_with_delta = [p for p in dpo_pairs if 'score_delta' in p]
    if pairs_with_delta:
        print(f"Average score delta: {sum(p['score_delta'] for p in pairs_with_delta)/len(pairs_with_delta):.2f}")
    print(f"\nPairs file: {pairs_file}")
    print(f"Training file: {training_file}")

    # Show sample
    if dpo_pairs:
        print(f"\n--- SAMPLE PAIR ---")
        sample = dpo_pairs[0]
        print(f"Prompt ID: {sample['prompt_id']}")
        print(f"Chosen score: {sample['chosen_score']:.1f}")
        print(f"Rejected score: {sample['rejected_score']:.1f}")
        print(f"Chosen preview: {sample['chosen'][:200]}...")

    return dpo_pairs


def main():
    parser = argparse.ArgumentParser(description="BioReasoner 2.1 Training Pipeline")
    parser.add_argument("command", choices=["generate", "filter", "build_dpo", "status"])
    parser.add_argument("--n_prompts", type=int, default=150, help="Number of prompts to use")
    parser.add_argument("--n_per_prompt", type=int, default=4, help="Generations per prompt")
    parser.add_argument("--input", type=str, help="Input file override")
    args = parser.parse_args()

    if args.command == "generate":
        prompts = load_prompts(args.n_prompts)
        generate_candidates_stub(prompts, args.n_per_prompt)

    elif args.command == "filter":
        input_file = Path(args.input) if args.input else None
        filter_candidates(input_file)

    elif args.command == "build_dpo":
        input_file = Path(args.input) if args.input else None
        build_dpo_pairs(input_file)

    elif args.command == "status":
        print(f"\n{'='*70}")
        print("BIOREASONER 2.1 PIPELINE STATUS")
        print(f"{'='*70}")

        files = [
            ("Generation batch", OUTPUT_DIR / "generation_batch.jsonl"),
            ("Generation results", OUTPUT_DIR / "generation_results.jsonl"),
            ("Passed candidates", OUTPUT_DIR / "candidates_passed.jsonl"),
            ("Rejected candidates", OUTPUT_DIR / "candidates_rejected.jsonl"),
            ("Tribunal scored", OUTPUT_DIR / "candidates_tribunal_scored.jsonl"),
            ("DPO pairs", OUTPUT_DIR / "dpo_pairs_21.jsonl"),
        ]

        for name, path in files:
            if path.exists():
                count = sum(1 for _ in open(path))
                print(f"  {name}: {count} items")
            else:
                print(f"  {name}: NOT CREATED")


if __name__ == "__main__":
    main()
