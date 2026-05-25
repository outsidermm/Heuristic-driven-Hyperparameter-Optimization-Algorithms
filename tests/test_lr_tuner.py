"""Tests for LrTuner search logic.

Searcher.training() and DataLoader.load_dataset() are mocked so no GPU or
dataset files are needed.  Requires the ``tf`` optional extra.
"""

import math

import pytest

pytest.importorskip("tensorflow", reason="tensorflow not installed — run: uv sync --extra tf")

from algorithm.lr import LrTuner


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


def make_tuner(left=1, right=5, allowance=0.05):
    return LrTuner("cifar100", left_bound=left, right_bound=right,
                   local_extrema_allowance=allowance)


class TestLrTunerSearch:
    def test_returns_three_values(self, mocker):
        mocker.patch(
            "algorithm.searcher.Searcher.training",
            return_value=(5.0, 0.8),
        )
        tuner = make_tuner()
        result = tuner.search()
        assert len(result) == 3

    def test_finds_local_maximum(self, mocker):
        """Mid accuracy clearly exceeds left and right → mid should be returned."""
        def training_by_lr(**kwargs):
            lr = kwargs.get("lr", 0.01)
            # mid exponent is 3 → lr=1e-3 → high accuracy
            if abs(lr - 1e-3) < 1e-9:
                return (5.0, 0.90)
            return (5.0, 0.50)

        mocker.patch(
            "algorithm.searcher.Searcher.training",
            side_effect=lambda **kw: training_by_lr(**kw),
        )
        tuner = make_tuner(left=1, right=5, allowance=0.05)
        lr_exp, acc, _ = tuner.search()
        # Should find the exponent whose lr=1e-3 (exponent=3)
        assert lr_exp == 3
        assert acc == pytest.approx(0.90)

    def test_no_local_max_returns_sentinel(self, mocker):
        """Monotonically increasing accuracy → no local max → return (-1,-1,-1)."""
        def monotone(**kwargs):
            lr = kwargs.get("lr", 0.01)
            # Higher lr (smaller exponent) → better accuracy; no local maximum
            exp = -math.log10(lr) if lr > 0 else 7
            return (5.0, 1.0 - exp * 0.1)

        mocker.patch(
            "algorithm.searcher.Searcher.training",
            side_effect=lambda **kw: monotone(**kw),
        )
        tuner = make_tuner(left=1, right=3, allowance=0.50)
        result = tuner.search()
        assert result == (-1, -1, -1)

    def test_caches_training_calls(self, mocker):
        """The same lr_linear value should only trigger one Searcher.training call."""
        training_mock = mocker.patch(
            "algorithm.searcher.Searcher.training",
            return_value=(5.0, 0.8),
        )
        tuner = make_tuner(left=2, right=2, allowance=0.05)
        tuner.data_stat_search(2)
        tuner.data_stat_search(2)  # second call should hit cache
        assert training_mock.call_count == 1
