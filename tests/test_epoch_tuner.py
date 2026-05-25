"""Tests for EpochTuner binary search logic.

Searcher.training() and DataLoader.load_dataset() are mocked so no GPU or
dataset files are needed.  Requires the ``tf`` optional extra (TensorFlow
is imported at module level in algorithm/).
"""

import pytest

pytest.importorskip("tensorflow", reason="tensorflow not installed — run: uv sync --extra tf")

from algorithm.epoch import EpochTuner


@pytest.fixture(autouse=True)
def _no_dataloader(mocker):
    mocker.patch(
        "utility.dataloader.DataLoader.load_dataset",
        return_value=(None, None, None),
    )


@pytest.fixture(autouse=True)
def _no_tf_strategy(mocker):
    """Prevent MirroredStrategy from requiring a GPU."""
    strategy = mocker.MagicMock()
    strategy.num_replicas_in_sync = 1
    mocker.patch("tensorflow.distribute.MirroredStrategy", return_value=strategy)


def make_tuner(left=1, right=8, exploration_factor=1):
    return EpochTuner("cifar100", left_bound=left, right_bound=right,
                      exploration_factor=exploration_factor)


class TestEpochTunerWeightedAvg:
    """weighted_avg_from_epoch uses the cached epoch/accuracy/time lists."""

    def test_returns_midpoint_for_equal_weights(self, mocker):
        mocker.patch(
            "algorithm.searcher.Searcher.training",
            side_effect=[(10.0, 0.5), (20.0, 0.8)],
        )
        tuner = make_tuner(left=1, right=2)
        tuner.epoch_run(1)
        tuner.epoch_run(2)
        # After two runs, normalized time=[0,1], accuracy=[0,1]
        # weighted_avg for epoch 1: time_norm=0, (1-acc_norm)=1 → avg=0.5
        result = tuner.weighted_avg_from_epoch(1)
        assert result == pytest.approx(0.5)

    def test_more_efficient_epoch_has_lower_score(self, mocker):
        mocker.patch(
            "algorithm.searcher.Searcher.training",
            side_effect=[(10.0, 0.9), (20.0, 0.5)],
        )
        tuner = make_tuner(left=1, right=2)
        tuner.epoch_run(1)
        tuner.epoch_run(2)
        # epoch 1: fast + high accuracy → lower score (better)
        assert tuner.weighted_avg_from_epoch(1) < tuner.weighted_avg_from_epoch(2)


class TestEpochTunerBinarySearch:
    def test_returns_tuple_of_three(self, mocker):
        # Provide enough return values for multiple training calls
        mocker.patch(
            "algorithm.searcher.Searcher.training",
            return_value=(5.0, 0.8),
        )
        tuner = make_tuner(left=2, right=4, exploration_factor=2)
        result = tuner.binary_search_efficient_epoch()
        assert len(result) == 3

    def test_best_epoch_within_bounds(self, mocker):
        mocker.patch(
            "algorithm.searcher.Searcher.training",
            return_value=(5.0, 0.8),
        )
        tuner = make_tuner(left=2, right=4, exploration_factor=2)
        best_epoch, acc, elapsed = tuner.binary_search_efficient_epoch()
        assert 2 <= best_epoch <= 4
        assert 0.0 <= acc <= 1.0
        assert elapsed >= 0.0

    def test_prefers_faster_high_accuracy_epoch(self, mocker):
        """When left is fast+accurate and right is slow+low, left should win."""
        # epoch_run is called with left=1 and right=4 first, then mid=2
        # We want epoch 1 to look best (fast, high acc)
        def training_side_effect(**kwargs):
            epoch = kwargs.get("epoch", 250)
            if epoch == 1:
                return (1.0, 0.9)
            return (100.0, 0.1)

        mocker.patch(
            "algorithm.searcher.Searcher.training",
            side_effect=lambda **kw: training_side_effect(**kw),
        )
        tuner = make_tuner(left=1, right=4, exploration_factor=2)
        best_epoch, acc, elapsed = tuner.binary_search_efficient_epoch()
        # epoch 1 should have been selected as most efficient
        assert best_epoch == 1
