"""The SLURM grid: the submitting shell and each worker must enumerate it identically."""

import argparse

from tfmdm.cli import _groups


def _args(**kwargs):
    defaults = {"datasets": ["adult", "taiwan"], "models": ["ebm", "nam"],
                "arms": ["hard", "distilled"], "chunk": None, "split_seeds": [0, 1]}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_group_count_is_the_product_of_the_dimensions():
    groups = _groups(_args())
    assert len(groups) == 2 * 2 * 2 * 2  # splits x datasets x models x arms


def test_every_group_carries_all_run_seeds_by_default():
    for group in _groups(_args()):
        assert group["seeds"] == list(range(30))


def test_enumeration_is_deterministic():
    """The array index IS the grid coordinate, so two calls must agree exactly."""
    assert _groups(_args()) == _groups(_args())


def test_chunking_partitions_the_seeds_without_loss_or_overlap():
    groups = _groups(_args(chunk=10, split_seeds=[0], datasets=["adult"],
                           models=["ebm"], arms=["hard"]))
    assert len(groups) == 3
    seen = [s for g in groups for s in g["seeds"]]
    assert seen == list(range(30))


def test_chunking_handles_a_ragged_final_chunk():
    groups = _groups(_args(chunk=7, split_seeds=[0], datasets=["adult"],
                           models=["ebm"], arms=["hard"]))
    assert [len(g["seeds"]) for g in groups] == [7, 7, 7, 7, 2]


def test_split_seed_is_carried_on_every_group():
    groups = _groups(_args(split_seeds=[3, 4]))
    assert {g["split_seed"] for g in groups} == {3, 4}
