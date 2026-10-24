"""Repeatable, paired evaluation on boards reserved for experiment studies.

Use validation boards to choose settings/checkpoints. Choose the finalist before
opening the test split; repeatedly selecting on test results makes it validation.
Training draws below 2**31, and rl.evaluate's original boards start at 10 billion.
This module reserves much higher, separate blocks for validation and test.

    python -m rl.experiment_eval --model models/example-best.pt --split validation --games 2000 --output experiments/example-validation.json
    python -m rl.experiment_eval --model models/finalist-best.pt --split test --games 10000 --output experiments/finalist-test.json

Every result contains the complete per-board records, exact paired win tests,
a deterministic paired-bootstrap interval, checkpoint hashes, source hashes and
runtime configuration. Bayes is the current full bayes_reveal policy, including
its zero-opening tie break. Its per-board results are cached by protocol/source.
P-values describe one comparison and do not correct a search over many models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SPLIT_BASE = {"validation": 1 << 40, "test": 1 << 41}
LEVEL_STRIDE = 1 << 32
REPLICATE_STRIDE = 1 << 22
MAX_GAMES = 1_000_000
SOURCE_FILES = (
    "_native.py", "engine.py", "ai_agents.py", "cpp/msw/engine.hpp",
    "cpp/msw/solver.hpp", "rl/minesweeper_env.py", "rl/qnet.py",
    "rl/evaluate.py", "rl/baseline.py", "rl/experiment_eval.py",
    "rl/dqn.py", "rl/per.py", "rl/journal.py",
)
BAYES_SOURCE_FILES = (
    "_native.py", "engine.py", "ai_agents.py", "cpp/msw/engine.hpp",
    "cpp/msw/solver.hpp", "rl/minesweeper_env.py", "rl/baseline.py",
)


def board_seeds(split: str, level: int, games: int, replicate: int = 0) -> range:
    """A fixed block, independent of model/training seed and evaluation count.

    Replicates are independent board sets, not training seeds. A million boards
    leave room for the engine's automatic-first-click redeal within each block.
    """
    if split not in SPLIT_BASE:
        raise ValueError("split must be validation or test")
    if level not in (1, 2, 3, 4):
        raise ValueError("level must be 1, 2, 3 or 4")
    if not 1 <= games <= MAX_GAMES:
        raise ValueError(f"games must be between 1 and {MAX_GAMES}")
    if not 0 <= replicate < LEVEL_STRIDE // REPLICATE_STRIDE:
        raise ValueError("replicate must be between 0 and 1023")
    start = SPLIT_BASE[split] + level * LEVEL_STRIDE + replicate * REPLICATE_STRIDE
    return range(start, start + games)


def paired_comparison(model_records: list[dict], bayes_records: list[dict], *,
                      bootstrap_samples: int = 20_000, seed: int = 20261024) -> dict:
    """Compare wins on matching boards, retaining their pairing in the interval.

    The exact test is McNemar's conditional binomial test on discordant boards.
    The bootstrap resamples boards; it is not a confidence interval across
    independent training seeds. Repeat training to assess training variability.
    """
    from scipy.stats import binomtest

    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")

    def indexed(records):
        out = {int(r["seed"]): r for r in records}
        if len(out) != len(records):
            raise ValueError("duplicate effective board seeds; pairing is ambiguous")
        return out

    model, bayes = indexed(model_records), indexed(bayes_records)
    if not model or model.keys() != bayes.keys():
        raise ValueError("model and Bayes must contain the same nonempty set of board seeds")
    a = sum(bool(model[s]["won"]) and not bool(bayes[s]["won"]) for s in model)
    b = sum(bool(bayes[s]["won"]) and not bool(model[s]["won"]) for s in model)
    both = sum(bool(model[s]["won"]) and bool(bayes[s]["won"]) for s in model)
    n, discordant = len(model), a + b
    rng = np.random.default_rng(seed)
    # Multinomial counts exactly implement resampling the paired outcomes, while
    # avoiding an array with games * bootstrap_samples entries.
    samples = rng.multinomial(n, [a / n, b / n, (n - discordant) / n], size=bootstrap_samples)
    differences = (samples[:, 0] - samples[:, 1]) / n
    lo, hi = np.quantile(differences, [0.025, 0.975])
    return {
        "games": n,
        "model_only_wins": a,
        "bayes_only_wins": b,
        "both_win": both,
        "both_lose": n - discordant - both,
        "win_difference": (a - b) / n,
        "win_difference_ci95": [float(lo), float(hi)],
        "mcnemar_exact_two_sided_p": float(binomtest(a, discordant, 0.5).pvalue) if discordant else 1.0,
        "model_better_exact_one_sided_p": float(binomtest(a, discordant, 0.5, alternative="greater").pvalue)
        if discordant else 1.0,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {name: sha256(ROOT / name) for name in SOURCE_FILES}


def bayes_cache_protocol(protocol: dict) -> dict:
    """Only board selection and sources used by Bayes affect its cached games.

    Network/trainer/evaluator edits and inference precision/ensembling do not
    change Bayes. The complete source provenance stays in the result artifact.
    """
    keys = ("split", "level", "games", "replicate", "requested_seed_start", "requested_seed_stop")
    return {"schema": 1, **{key: protocol[key] for key in keys},
            "source_sha256": {name: protocol["source_sha256"][name] for name in BAYES_SOURCE_FILES}}


def bayes_cache_key(protocol: dict) -> str:
    return hashlib.sha256(json.dumps(bayes_cache_protocol(protocol), sort_keys=True).encode()).hexdigest()[:20]


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(_json_safe(data), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def evaluate(args) -> dict:
    # Keep statistical utilities usable without loading CUDA, cppyy or the game.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    from rl.baseline import play_bayes
    from rl.evaluate import format_row, play_q, summarize
    from rl.qnet import MODELS_DIR, SymmetryEnsemble, load_checkpoint

    seeds = board_seeds(args.split, args.level, args.games, args.replicate)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    hashes = source_hashes()
    protocol = {"schema": 1, "split": args.split, "level": args.level, "games": args.games,
                "replicate": args.replicate, "requested_seed_start": seeds.start,
                "requested_seed_stop": seeds.stop, "source_sha256": hashes}
    cache_protocol = bayes_cache_protocol(protocol)
    cache_key = bayes_cache_key(protocol)
    cache_dir = args.cache_dir or args.output.parent / "bayes-cache"
    cache = cache_dir / f"bayes-{args.split}-L{args.level}-{cache_key}.json"
    if cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if cached["protocol"] != cache_protocol:
            raise ValueError("Bayes cache protocol does not match")
        bayes_records = cached["records"]
    else:
        bayes_records = sorted(play_bayes(args.level, seeds, n_jobs=args.jobs), key=lambda r: r["seed"])
        write_json(cache, {"protocol": cache_protocol, "records": bayes_records})
    bayes_summary = summarize(bayes_records)
    print(format_row("bayes", bayes_summary), flush=True)
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    result = {
        "protocol": protocol,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_revision": revision,
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "numpy": np.__version__, "device": str(device),
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "bf16": device.type == "cuda" and not args.no_bf16,
                    "symmetries": args.symmetries, "fp32_head_override": args.fp32_head,
                    "deterministic_algorithms": True, "n_envs": args.n_envs},
        "interpretation": "Choose the finalist on validation before inspecting test. Paired intervals measure board uncertainty, not training-seed variation. P-values are unadjusted for multiple comparisons.",
        "bayes": {"summary": bayes_summary, "records": bayes_records, "cache": str(cache.resolve())},
        "models": [],
    }
    for name in args.model:
        path = Path(name)
        if not path.exists():
            path = MODELS_DIR / (path.name if path.suffix == ".pt" else f"{path.name}.pt")
        before_hash = sha256(path)
        q, checkpoint = load_checkpoint(path, device)
        if args.fp32_head:
            q.config["fp32_head"] = True
        inference_config = dict(q.config)
        if args.symmetries:
            q = SymmetryEnsemble(q, probabilities=(checkpoint.get("args") or {}).get("loss") == "bce")
        records = sorted(play_q(q, args.level, seeds, device, n_envs=args.n_envs,
                                breakdown=not args.no_breakdown, canvas=checkpoint.get("canvas"),
                                bf16=not args.no_bf16), key=lambda r: r["seed"])
        after_hash = sha256(path)
        if before_hash != after_hash:
            raise RuntimeError(f"checkpoint changed during evaluation: {path}; evaluate an immutable copy")
        summary = summarize(records)
        comparison = paired_comparison(records, bayes_records, bootstrap_samples=args.bootstrap_samples,
                                       seed=args.statistics_seed)
        result["models"].append({"path": str(path.resolve()), "sha256": before_hash,
                                 "training_step": checkpoint.get("global_step"),
                                 "training_args": checkpoint.get("args"), "config": checkpoint.get("config"),
                                 "inference_config": inference_config,
                                 "summary": summary, "paired_vs_bayes": comparison, "records": records})
        print(format_row(path.stem, summary), flush=True)
        print(f"  paired win difference {comparison['win_difference'] * 100:+.2f} points; "
              f"95% CI [{comparison['win_difference_ci95'][0] * 100:+.2f}, "
              f"{comparison['win_difference_ci95'][1] * 100:+.2f}]; "
              f"exact two-sided p={comparison['mcnemar_exact_two_sided_p']:.4g}", flush=True)
        write_json(args.output, result)  # retain completed models if a later one fails
        del q
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", nargs="+", required=True, help="one or more immutable checkpoint files")
    ap.add_argument("--split", choices=tuple(SPLIT_BASE), default="validation")
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--replicate", type=int, default=0, help="independent board block, not training seed")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    ap.add_argument("--n-envs", type=int, default=256)
    ap.add_argument("--jobs", type=int, default=8, help="Bayes worker processes")
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--symmetries", action="store_true", help="average rotated/reflected predictions (probabilities for BCE)")
    ap.add_argument("--fp32-head", action="store_true", help="override the loaded network to run its output head in FP32")
    ap.add_argument("--no-breakdown", action="store_true")
    ap.add_argument("--bootstrap-samples", type=int, default=20_000)
    ap.add_argument("--statistics-seed", type=int, default=20261024)
    args = ap.parse_args()
    if args.n_envs < 1 or args.jobs < 1:
        ap.error("--n-envs and --jobs must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
