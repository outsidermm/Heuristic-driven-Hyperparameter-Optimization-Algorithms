"""Tests for BatchSizeTuner search logic.

Searcher.training() and DataLoader.load_dataset() are mocked so no GPU or
dataset files are needed.  Requires the ``tf`` optional extra.
"""

import pytest

pytest.importorskip("tensorflow", reason="tensorflow not installed — run: uv sync --extra tf")

from algorithm.batch_size import BatchSizeTuner


@pytest.fixture(autouse=True)
def _no_dataloader(mocker):
    mocker.patch(
        "utility.dataloader.DataLoader.load_dataset",
        return_value=(None, None, None),
    )


@pytest.fixture(autouse=True)
def _no_tf_strategy(mocker):
    strategy = mocker.MagicMock()
    strategy.num_replicas_in_sync = 1
    mocker.patch("tensorflow.distribute.MirroredStrategy", return_value=strategy)


def make_tuner(left=4, right=6, acceptable_range=0.30):
    return BatchSizeTuner("cifar100", left_bound=left, right_bound=right,
                          acceptable_range=acceptable_range)


class TestBatchSizeTunerSearch:
    def test_returns_three_values(self, mocker):
        mocker.patch(
            "algorithm.searcher.Searcher.training",
            return_value=(5.0, 0.8),
        )
        tuner = make_tuner()
        result = tuner.search()
        assert len(result) == 3

    def test_best_batch_within_bounds(self, mocker):
        mocker.patch(
            "algorithm.searcher.Searcher.training",
            return_value=(5.0, 0.8),
        )
        tuner = make_tuner(left=4, right=6)
        best_batch, acc, elapsed = tuner.search()
        # batch size is 2**left to 2**right (16 to 64)
        assert 2**4 <= best_batch <= 2**6

    def test_oom_at_minimum_returns_sentinel(self, mocker):
        """ResourceExhaustedError at left_bound → returns (-1, -1, -1)."""
        import tensorflow as tf

        mocker.patch(
            "algorithm.searcher.Searcher.training",
            side_effect=tf.errors.ResourceExhaustedError(None, None, "OOM"),
        )
        tuner = make_tuner()
        result = tuner.search()
        assert result == (-1.0, -1.0, -1.0)

    def test_accuracy_stays_within_range(self, mocker):
        """If all accuracies are identical, the largest batch should be chosen."""
        mocker.patch(
            "algorithm.searcher.Searcher.training",
            return_value=(3.0, 0.75),
        )
        tuner = make_tuner(left=4, right=5, acceptable_range=0.50)
        best_batch, acc, _ = tuner.search()
        assert acc == pytest.approx(0.75)
