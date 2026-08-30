"""Split replicates must be unable to collide on disk."""

from tfmdm import paths


def test_artifacts_are_namespaced_by_split_seed():
    a = paths.preds("adult", "ebm", "distilled", 7, split_seed=0)
    b = paths.preds("adult", "ebm", "distilled", 7, split_seed=1)
    assert a != b
    assert a.name == b.name           # same filename...
    assert a.parent != b.parent       # ...different replicate directory


def test_every_split_dependent_artifact_lives_under_its_split_root():
    root = paths.split_root(2)
    for path in (
        paths.splits("adult", 2),
        paths.view("adult", "encoded", 2),
        paths.transformer("adult", 2),
        paths.soft_train("adult", 2),
        paths.soft_val("adult", 2),
        paths.preds("adult", "ebm", "hard", 0, 2),
        paths.importances("adult", "ebm", "hard", 0, 2),
    ):
        assert root in path.parents, path


def test_cleaned_frame_is_shared_across_splits():
    """Cleaning happens before any split exists, so it must not be duplicated."""
    assert paths.processed("adult") == paths.processed("adult")
    assert "split" not in paths.processed("adult").parent.name


def test_tuned_configs_stay_under_configs_so_they_can_be_committed():
    path = paths.tuned_config("adult", "ebm", "hard", 3)
    assert paths.CONFIGS in path.parents
    assert path.parent.name == "split3"
