"""Single entry point. Every Taskfile target and SLURM job is one subcommand here."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

DATASETS = ["adult", "taiwan"]
INTERPRETABLE = ["ebm", "nam"]
ARMS = ["hard", "distilled"]


def _allow_dirty() -> bool:
    return os.environ.get("ALLOW_DIRTY", "0") == "1"


def _emit(payload: object) -> None:
    print(json.dumps(payload, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tfmdm", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data", help="Phase 1: clean, split, build feature views")
    p.add_argument("--dataset", required=True, choices=DATASETS)

    p = sub.add_parser("probe", help="Phase 2.1: TabICLv2 feasibility probe")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--fraction", type=float, default=0.05)

    p = sub.add_parser("softlabels", help="Phase 2.2-2.3: cross-fitted TabICLv2 probabilities")
    p.add_argument("--dataset", required=True, choices=DATASETS)

    p = sub.add_parser("tabicl-preds", help="Phase 2.4: the TabICLv2 model set (baseline B3)")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("sanity", help="Phase 3: degenerate and reproduction gates")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--models", nargs="+", default=INTERPRETABLE)

    p = sub.add_parser("tune", help="Phase 4.2: one random search per (dataset, model, arm)")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--model", required=True, choices=INTERPRETABLE + ["logreg"])
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--n-configs", type=int, default=None)

    p = sub.add_parser("train", help="Phase 4.1: one sweep cell")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--model", required=True, choices=INTERPRETABLE + ["logreg"])
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("sweep", help="Run many training cells locally (SLURM does this in parallel)")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--models", nargs="+", default=INTERPRETABLE)
    p.add_argument("--arms", nargs="+", default=ARMS)
    p.add_argument("--seeds", nargs="+", type=int, default=None)

    p = sub.add_parser("cells", help="Print the sweep grid as JSON lines (drives the SLURM array)")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--models", nargs="+", default=INTERPRETABLE)
    p.add_argument("--arms", nargs="+", default=ARMS)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--index", type=int, default=None, help="Print only this row of the grid")

    p = sub.add_parser("analyze", help="Phase 5.1-5.3: metrics, intervals, comparisons")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--models", nargs="+", default=INTERPRETABLE + ["tabicl"])
    p.add_argument("--arms", nargs="+", default=ARMS + ["incontext"])

    p = sub.add_parser("figures", help="Render F1-F4")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--models", nargs="+", default=INTERPRETABLE)

    return parser


def _grid(args) -> list[dict]:
    from .config import load

    cells = []
    for dataset in args.datasets:
        seed_list = args.seeds or [int(s) for s in load(dataset).seeds]
        for model in args.models:
            for arm in args.arms:
                for seed in seed_list:
                    cells.append({"dataset": dataset, "model": model, "arm": arm, "seed": seed})
    return cells


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command

    if command == "data":
        from .stages import data

        _emit(data.run(args.dataset))

    elif command == "probe":
        from .stages import tabicl_preds

        _emit(tabicl_preds.probe(args.dataset, args.fraction))

    elif command == "softlabels":
        from .stages import softlabels

        _emit(softlabels.run(args.dataset, allow_dirty=_allow_dirty()))

    elif command == "tabicl-preds":
        from .stages import tabicl_preds

        _emit(tabicl_preds.run(args.dataset, args.seed, allow_dirty=_allow_dirty(),
                               overwrite=args.overwrite))

    elif command == "sanity":
        from .stages import sanity

        report = sanity.run(args.datasets, args.models)
        _emit(report)
        return 0 if report["all_passed"] else 1

    elif command == "tune":
        from .stages import tune

        _emit(tune.run(args.dataset, args.model, args.arm, args.n_configs))

    elif command == "train":
        from .stages import train

        _emit(train.run(args.dataset, args.model, args.arm, args.seed,
                        allow_dirty=_allow_dirty(), overwrite=args.overwrite))

    elif command == "sweep":
        from .stages import train

        results = [train.run(c["dataset"], c["model"], c["arm"], c["seed"],
                             allow_dirty=_allow_dirty())
                   for c in _grid(args)]
        _emit({"n_cells": len(results),
               "skipped": sum(r["status"] == "skipped" for r in results)})

    elif command == "cells":
        cells = _grid(args)
        if args.index is not None:
            if args.index >= len(cells):
                print(f"Index {args.index} beyond grid of {len(cells)} cells", file=sys.stderr)
                return 1
            print(json.dumps(cells[args.index]))
        else:
            for cell in cells:
                print(json.dumps(cell))

    elif command == "analyze":
        from .analysis import aggregate

        result = aggregate(args.datasets, args.models, args.arms)
        _emit({"n_summaries": len(result["summaries"]),
               "n_comparisons": len(result["comparisons"])})

    elif command == "figures":
        from .analysis import figures

        _emit(figures.run(args.datasets, args.models))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
