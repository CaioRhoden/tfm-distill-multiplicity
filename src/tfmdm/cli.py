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


def _add_split(parser: argparse.ArgumentParser, plural: bool = False) -> None:
    """Attach the split-seed selector.

    Defaults come from `split.seed` / `split.seeds` in configs/base.yaml, so every
    command works unchanged on the primary split and opts into the replicates.
    """
    if plural:
        parser.add_argument("--split-seeds", nargs="+", type=int, default=None,
                            help="Split replicates to cover (default: split.seeds)")
    else:
        parser.add_argument("--split-seed", type=int, default=None,
                            help="Split replicate to act on (default: split.seed)")


def _resolve_split(value: int | None) -> int:
    if value is not None:
        return int(value)
    from .config import load

    return int(load(DATASETS[0]).split.seed)


def _resolve_splits(values: Sequence[int] | None) -> list[int]:
    if values:
        return [int(v) for v in values]
    from .config import load, split_seeds

    return split_seeds(load(DATASETS[0]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tfmdm", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data", help="Phase 1: clean, split, build feature views")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    _add_split(p)

    p = sub.add_parser("probe", help="Phase 2.1: TabICLv2 feasibility probe")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--fraction", type=float, default=0.05)
    _add_split(p)

    p = sub.add_parser("softlabels", help="Phase 2.2-2.3: cross-fitted TabICLv2 probabilities")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    _add_split(p)

    p = sub.add_parser("tabicl-preds", help="Phase 2.4: the TabICLv2 model set (baseline B3)")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--overwrite", action="store_true")
    _add_split(p)

    p = sub.add_parser("sanity", help="Phase 3: degenerate and reproduction gates")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--models", nargs="+", default=INTERPRETABLE)
    _add_split(p)

    p = sub.add_parser("tune", help="Phase 4.2: one random search per (dataset, model, arm, split)")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--model", required=True, choices=INTERPRETABLE + ["logreg"])
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--n-configs", type=int, default=None)
    _add_split(p)

    p = sub.add_parser("train", help="Phase 4.1: one cell (single seed), for debugging")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--model", required=True, choices=INTERPRETABLE + ["logreg"])
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--overwrite", action="store_true")
    _add_split(p)

    p = sub.add_parser("train-group",
                       help="Phase 4.1: every run seed of one (dataset, model, arm, split)")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--model", required=True, choices=INTERPRETABLE + ["logreg"])
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Run seeds (default: all of model_seeds)")
    p.add_argument("--overwrite", action="store_true")
    _add_split(p)

    p = sub.add_parser("groups", help="Print the group grid, one JSON object per line")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--models", nargs="+", default=INTERPRETABLE)
    p.add_argument("--arms", nargs="+", default=ARMS)
    p.add_argument("--chunk", type=int, default=None,
                   help="Split each group's run seeds into chunks of this size")
    p.add_argument("--index", type=int, default=None, help="Print only this row of the grid")
    _add_split(p, plural=True)

    p = sub.add_parser("sweep", help="Run every group locally (SLURM does this in parallel)")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--models", nargs="+", default=INTERPRETABLE)
    p.add_argument("--arms", nargs="+", default=ARMS)
    p.add_argument("--chunk", type=int, default=None)
    _add_split(p, plural=True)

    p = sub.add_parser("analyze", help="Phase 5.1-5.3: metrics, intervals, comparisons")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--models", nargs="+", default=INTERPRETABLE + ["tabicl"])
    p.add_argument("--arms", nargs="+", default=ARMS + ["incontext"])
    _add_split(p)

    p = sub.add_parser("explanations",
                       help="Phase 5.4: explanation multiplicity over the fitted model sets")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--models", nargs="+", default=INTERPRETABLE)
    p.add_argument("--arms", nargs="+", default=ARMS)
    p.add_argument("--max-rows", type=int, default=None,
                   help="Explain a random subsample of this many test rows (default: all)")
    _add_split(p)

    p = sub.add_parser("compile-explanations",
                       help="Pool explanation multiplicity and join it to AUROC")
    _add_split(p, plural=True)

    p = sub.add_parser("combine",
                       help="Pool per-split results into results/all_seed_metrics.csv "
                            "and results/all_arm_summaries.csv")
    _add_split(p, plural=True)

    p = sub.add_parser("figures", help="Render F1-F4 for one split")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--models", nargs="+", default=INTERPRETABLE)
    _add_split(p)

    return parser


def _groups(args) -> list[dict]:
    """The unit of SLURM work: one (dataset, model, arm, split) and its run seeds.

    A group is one job because the expensive part of a cell is loading the view, the
    split and the soft labels, which all 30 seeds share. ``--chunk`` trades that saving
    back for parallelism when the queue has room.
    """
    from .config import load

    groups = []
    for split_seed in _resolve_splits(getattr(args, "split_seeds", None)):
        for dataset in args.datasets:
            run_seeds = [int(s) for s in load(dataset).model_seeds]
            chunks = ([run_seeds] if not args.chunk
                      else [run_seeds[i:i + args.chunk]
                            for i in range(0, len(run_seeds), args.chunk)])
            for model in args.models:
                for arm in args.arms:
                    for chunk in chunks:
                        groups.append({"dataset": dataset, "model": model, "arm": arm,
                                       "split_seed": split_seed, "seeds": chunk})
    return groups


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command

    if command == "data":
        from .stages import data

        _emit(data.run(args.dataset, _resolve_split(args.split_seed)))

    elif command == "probe":
        from .stages import tabicl_preds

        _emit(tabicl_preds.probe(args.dataset, _resolve_split(args.split_seed), args.fraction))

    elif command == "softlabels":
        from .stages import softlabels

        _emit(softlabels.run(args.dataset, _resolve_split(args.split_seed),
                             allow_dirty=_allow_dirty()))

    elif command == "tabicl-preds":
        from .stages import tabicl_preds

        _emit(tabicl_preds.run(args.dataset, args.seed, _resolve_split(args.split_seed),
                               allow_dirty=_allow_dirty(), overwrite=args.overwrite))

    elif command == "sanity":
        from .stages import sanity

        report = sanity.run(args.datasets, args.models, _resolve_split(args.split_seed))
        _emit(report)
        return 0 if report["all_passed"] else 1

    elif command == "tune":
        from .stages import tune

        _emit(tune.run(args.dataset, args.model, args.arm,
                       _resolve_split(args.split_seed), args.n_configs))

    elif command == "train":
        from .stages import train

        _emit(train.run(args.dataset, args.model, args.arm, args.seed,
                        _resolve_split(args.split_seed),
                        allow_dirty=_allow_dirty(), overwrite=args.overwrite))

    elif command == "train-group":
        from .stages import train

        _emit(train.run_group(args.dataset, args.model, args.arm,
                              _resolve_split(args.split_seed), args.seeds,
                              allow_dirty=_allow_dirty(), overwrite=args.overwrite))

    elif command == "groups":
        groups = _groups(args)
        if args.index is not None:
            if args.index >= len(groups):
                print(f"Index {args.index} beyond grid of {len(groups)} groups", file=sys.stderr)
                return 1
            print(json.dumps(groups[args.index]))
        else:
            for group in groups:
                print(json.dumps(group))

    elif command == "sweep":
        from .stages import train

        results = [
            train.run_group(g["dataset"], g["model"], g["arm"], g["split_seed"], g["seeds"],
                            allow_dirty=_allow_dirty())
            for g in _groups(args)
        ]
        _emit({"n_groups": len(results),
               "trained": sum(r.get("trained", 0) for r in results),
               "skipped_groups": sum(r["status"] == "skipped" for r in results)})

    elif command == "analyze":
        from .analysis import aggregate

        result = aggregate(args.datasets, args.models, args.arms,
                           _resolve_split(args.split_seed))
        _emit({"n_summaries": len(result["summaries"]),
               "n_comparisons": len(result["comparisons"])})

    elif command == "explanations":
        from .analysis import explanations

        _emit(explanations.run(args.datasets, args.models, args.arms,
                               _resolve_split(args.split_seed), args.max_rows))

    elif command == "compile-explanations":
        from .analysis import explanations

        _emit(explanations.combine(_resolve_splits(args.split_seeds)))

    elif command == "combine":
        from .analysis import combine

        _emit(combine(_resolve_splits(args.split_seeds)))

    elif command == "figures":
        from .analysis import figures

        _emit(figures.run(args.datasets, args.models, _resolve_split(args.split_seed)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
