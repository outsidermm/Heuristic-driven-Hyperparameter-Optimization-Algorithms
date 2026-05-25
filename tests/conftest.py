"""Shared pytest fixtures."""

import pytest


@pytest.fixture
def mock_searcher_training(mocker):
    """Patch Searcher so it never touches TF/GPU.

    Returns a factory: call it with a list of (time, accuracy) pairs that will
    be returned in order on successive Searcher().training() calls.
    """

    def _factory(return_values: list[tuple[float, float]]):
        side_effects = list(return_values)
        mock = mocker.patch("algorithm.searcher.Searcher.training", side_effect=side_effects)
        return mock

    return _factory


@pytest.fixture
def mock_dataloader(mocker):
    """Patch DataLoader.load_dataset so no .npy files are needed."""
    mocker.patch(
        "utility.dataloader.DataLoader.load_dataset",
        return_value=(None, None, None),
    )
