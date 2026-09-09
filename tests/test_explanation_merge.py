"""A run restricted to one family must not delete the other family's results.

``tfmdm explanations --models nam`` and ``--models ebm`` are routinely run separately --
they have very different costs, and one may be rerun after a fix while the other is
still valid. If the second run overwrote the table, the result would be a file that
looks complete and is missing half the study, which is the kind of loss nothing
downstream can detect.
"""

import pandas as pd

from tfmdm.analysis.explanations import CELL_KEYS, COMPARISON_KEYS, _merge_into


def _cell(model, arm="hard", ambiguity=0.5, **extra):
    return {"dataset": "adult", "model": model, "arm": arm, "split_seed": 0,
            "attribution_ambiguity": ambiguity, **extra}


def test_a_family_not_recomputed_is_preserved(tmp_path):
    path = tmp_path / "explanation_multiplicity.csv"
    pd.DataFrame([_cell("ebm"), _cell("nam")]).to_csv(path, index=False)

    merged = _merge_into(path, [_cell("nam", ambiguity=0.1)], CELL_KEYS)

    assert sorted(merged["model"]) == ["ebm", "nam"]
    assert float(merged.loc[merged["model"] == "nam", "attribution_ambiguity"].iloc[0]) == 0.1
    assert float(merged.loc[merged["model"] == "ebm", "attribution_ambiguity"].iloc[0]) == 0.5


def test_recomputing_a_cell_replaces_it_rather_than_duplicating_it(tmp_path):
    path = tmp_path / "cells.csv"
    pd.DataFrame([_cell("nam", ambiguity=0.5)]).to_csv(path, index=False)

    merged = _merge_into(path, [_cell("nam", ambiguity=0.9)], CELL_KEYS)

    assert len(merged) == 1
    assert float(merged["attribution_ambiguity"].iloc[0]) == 0.9


def test_arms_are_part_of_the_cell_key(tmp_path):
    path = tmp_path / "cells.csv"
    pd.DataFrame([_cell("nam", "hard"), _cell("nam", "distilled")]).to_csv(path, index=False)

    merged = _merge_into(path, [_cell("nam", "hard", ambiguity=0.2)], CELL_KEYS)

    assert len(merged) == 2
    assert float(merged.loc[merged["arm"] == "distilled", "attribution_ambiguity"].iloc[0]) == 0.5


def test_comparisons_are_keyed_without_the_arm_column(tmp_path):
    """E2 rows span two arms, so a family's comparisons are replaced as a block."""
    path = tmp_path / "comparisons.csv"
    rows = [{"dataset": "adult", "model": m, "split_seed": 0, "hypothesis": "E2",
             "metric": "attribution_ambiguity", "arm_a": "distilled", "arm_b": "hard",
             "delta_point": -0.3} for m in ("ebm", "nam")]
    pd.DataFrame(rows).to_csv(path, index=False)

    fresh = dict(rows[1], delta_point=-0.9)
    merged = _merge_into(path, [fresh], COMPARISON_KEYS)

    assert len(merged) == 2
    assert float(merged.loc[merged["model"] == "nam", "delta_point"].iloc[0]) == -0.9
    assert float(merged.loc[merged["model"] == "ebm", "delta_point"].iloc[0]) == -0.3


def test_rows_from_an_older_metric_set_are_dropped_not_merged(tmp_path):
    """Old-schema rows lack the new columns; keeping them would read as zero, not stale."""
    path = tmp_path / "cells.csv"
    pd.DataFrame([{"dataset": "adult", "model": "ebm", "arm": "hard", "split_seed": 0,
                   "mean_spearman_correlation": 0.77}]).to_csv(path, index=False)

    merged = _merge_into(path, [_cell("nam")], CELL_KEYS)

    assert list(merged["model"]) == ["nam"]
    assert "attribution_ambiguity" in merged.columns
    assert not merged["attribution_ambiguity"].isna().any()


def test_a_missing_file_is_simply_the_new_rows(tmp_path):
    merged = _merge_into(tmp_path / "absent.csv", [_cell("nam")], CELL_KEYS)
    assert len(merged) == 1


def test_no_new_rows_leaves_an_empty_frame(tmp_path):
    path = tmp_path / "cells.csv"
    pd.DataFrame([_cell("ebm")]).to_csv(path, index=False)
    assert _merge_into(path, [], CELL_KEYS).empty
