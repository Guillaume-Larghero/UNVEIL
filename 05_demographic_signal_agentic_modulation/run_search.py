import argparse
import asyncio
import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd
import yaml

import agentic_search as search

SUMMARY_METRICS = ["combined_score", "valid", "G", "H", "fitness", "mean_dose", "error"]


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def manifest(args, settings):
    files = [args.seed, args.config, search.HERE / "agentic_search.py", search.HERE / "tile_filter.py"]
    cells = settings.cells.assign(manifest_md5=[md5(p) for p in settings.cells["manifest"]])
    return {"seed": str(args.seed), "config": str(args.config), "iterations": args.iterations,
            "md5": {str(p): md5(p) for p in files}, "eps": settings.eps, "lam": settings.lam,
            "cells": cells.to_dict(orient="records")}


def load_candidates(out_dir):
    rows = {}
    checkpoints = sorted(Path(out_dir, "checkpoints").glob("checkpoint_*"), key=lambda p: int(re.sub(r"\D", "", p.name)))
    for checkpoint in checkpoints:
        for path in sorted((checkpoint / "programs").glob("*.json")):
            program = json.loads(path.read_text())
            metrics = program.get("metrics") or {}
            rows[program["id"]] = {"id": program["id"], "iteration": program.get("iteration_found"), "parent": program.get("parent_id"),
                                   **{key: metrics.get(key) for key in SUMMARY_METRICS}, "code": program.get("code")}
    return pd.DataFrame(rows.values())


def summarise(out_dir):
    candidates = load_candidates(out_dir)
    if candidates.empty:
        raise SystemExit(f"no candidate programs under {out_dir}/checkpoints")
    candidates = candidates.sort_values("combined_score", ascending=False, kind="stable")
    candidates.drop(columns="code").to_csv(Path(out_dir) / "candidates.csv", index=False)
    best = candidates.iloc[0]
    Path(out_dir, "best_program.py").write_text(best["code"])
    print(f"{len(candidates)} candidates; best {best['id']} G={best['G']} H={best['H']} fitness={best['combined_score']}")


def run(args, settings):
    from openevolve import OpenEvolve
    from openevolve.config import Config

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise SystemExit(f"{out} already holds a search; choose another --out")
    (out / "manifest.json").write_text(json.dumps(manifest(args, settings), indent=2, default=str))
    os.environ["UNVEIL_SEARCH_CONFIG"] = str(Path(args.config).resolve())
    config = yaml.safe_load(Path(args.config).read_text())
    config.pop("search")
    controller = OpenEvolve(initial_program_path=str(args.seed), evaluation_file=str(search.HERE / "agentic_search.py"),
                            config=Config.from_dict(config), output_dir=str(out))
    asyncio.run(controller.run(iterations=args.iterations))
    summarise(out)


def freeze_check(args, settings):
    drop_scores = search.load_program(args.score)
    summaries, tables = [], []
    for seed in args.control_seeds:
        summary, table = search.score_program(drop_scores, settings, role=args.role, control_seed=seed)
        summaries.append({"role": args.role, "control_seed": seed, **summary})
        tables.append(table)
    summaries = pd.DataFrame(summaries)
    print(summaries.to_string(index=False))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        pd.concat(tables).to_csv(args.output, index=False)
        summaries.to_csv(Path(args.output).with_suffix(".summary.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="LLM-guided search over tile-filter programs that remove demographic signal while sparing the pathology task.")
    parser.add_argument("--config", type=Path, default=search.DEFAULT_CONFIG)
    parser.add_argument("--seed", type=Path, default=search.SEED_PROGRAM, help="program the search starts from")
    parser.add_argument("--out", default=None, help="output directory of the search")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true", help="score the seed program on the development cells and stop")
    parser.add_argument("--summarise", default=None, help="write candidates.csv and best_program.py for an existing search directory")
    parser.add_argument("--score", type=Path, default=None, help="score one program, e.g. a frozen winner, instead of searching")
    parser.add_argument("--role", choices=["dev", "held_out"], default="held_out")
    parser.add_argument("--control-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", default=None, help="per-cell csv of --score")
    args = parser.parse_args()

    if args.summarise:
        return summarise(args.summarise)
    settings = search.load_settings(args.config)
    if args.score:
        return freeze_check(args, settings)
    if args.dry_run:
        summary, table = search.score_program(search.load_program(args.seed), settings)
        print(table.to_string(index=False))
        print(summary)
        return
    if not args.out:
        parser.error("--out is required to run a search")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("set OPENAI_API_KEY outside the repository, e.g. source ~/.config/unveil/llm.env")
    run(args, settings)


if __name__ == "__main__":
    main()
