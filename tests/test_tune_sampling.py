"""The random search must spend its budget on distinct configurations.

Sampling is with replacement, so on a small grid -- the linear family's is six points --
a plain loop of ``n_configs`` draws re-scores configurations it has already fitted and
reports the winner of a search that never covered the space. The dedupe is bounded, so
it is also checked that it terminates on a space smaller than the budget.
"""

import numpy as np

from tfmdm.stages.tune import _distinct_configs

SMALL = {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
LARGE = {"lr": [0.0003, 0.001, 0.003], "dropout": [0.0, 0.1, 0.2],
         "hidden_sizes": [[64, 32], [128, 64], [32]], "l2": [0.0, 0.001, 0.01]}


def _rng():
    return np.random.default_rng(1000)


def test_a_space_smaller_than_the_budget_is_covered_exactly_once():
    configs = _distinct_configs(SMALL, {"max_iter": 5000}, 20, _rng())
    assert len(configs) == len(SMALL["C"])
    assert sorted(c["C"] for c in configs) == sorted(SMALL["C"])


def test_base_params_are_carried_into_every_config():
    for config in _distinct_configs(SMALL, {"max_iter": 5000, "penalty": "l2"}, 20, _rng()):
        assert config["max_iter"] == 5000 and config["penalty"] == "l2"


def test_a_large_space_still_spends_the_whole_budget():
    """81 combinations against a budget of 20: no draw should be given up on."""
    assert len(_distinct_configs(LARGE, {}, 20, _rng())) == 20


def test_configs_are_unique_and_the_draw_is_reproducible():
    configs = _distinct_configs(LARGE, {}, 20, _rng())
    keys = [tuple(sorted((k, str(v)) for k, v in c.items())) for c in configs]
    assert len(set(keys)) == len(keys)
    assert configs == _distinct_configs(LARGE, {}, 20, _rng())
