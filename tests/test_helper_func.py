"""Tests for utility/helper_func.py — all pure functions, no mocking needed."""

import csv

import numpy as np
import pytest

from utility.helper_func import min_max_scalar, weighted_avg, write_csv, write_header


class TestMinMaxScalar:
    def test_range_is_zero_to_one(self):
        data = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = min_max_scalar(data)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_midpoint(self):
        data = np.array([0.0, 5.0, 10.0])
        result = min_max_scalar(data)
        assert result[1] == pytest.approx(0.5)

    def test_single_element_is_nan(self):
        # (x - x) / (x - x) = 0/0 → NaN; document this known edge case
        data = np.array([42.0])
        result = min_max_scalar(data)
        assert np.isnan(result[0])

    def test_preserves_array_length(self):
        data = np.array([1.0, 3.0, 5.0, 7.0])
        assert len(min_max_scalar(data)) == 4


class TestWeightedAvg:
    def test_equal_weights(self):
        assert weighted_avg(0.0, 1.0) == pytest.approx(0.5)

    def test_custom_weights(self):
        assert weighted_avg(1.0, 0.0, weight1=0.8, weight2=0.2) == pytest.approx(0.8)

    def test_same_values(self):
        assert weighted_avg(0.3, 0.3, 0.4, 0.6) == pytest.approx(0.3)

    def test_zero_and_one(self):
        assert weighted_avg(0.0, 0.0) == pytest.approx(0.0)
        assert weighted_avg(1.0, 1.0) == pytest.approx(1.0)


class TestWriteCsvAndHeader:
    def test_write_header_creates_row(self, tmp_path):
        csv_file = tmp_path / "out.csv"
        write_header(["a", "b", "c"], str(csv_file))
        rows = csv_file.read_text().splitlines()
        assert rows[0] == "a,b,c"

    def test_write_csv_appends_rows(self, tmp_path):
        csv_file = tmp_path / "out.csv"
        write_header(["x", "y"], str(csv_file))
        write_csv([{"x": 1, "y": 2}, {"x": 3, "y": 4}], ["x", "y"], str(csv_file))
        rows = list(csv.DictReader(csv_file.open()))
        assert len(rows) == 2
        assert rows[0]["x"] == "1"
        assert rows[1]["y"] == "4"

    def test_write_csv_multiple_calls_append(self, tmp_path):
        csv_file = tmp_path / "out.csv"
        write_header(["v"], str(csv_file))
        write_csv([{"v": 10}], ["v"], str(csv_file))
        write_csv([{"v": 20}], ["v"], str(csv_file))
        rows = list(csv.DictReader(csv_file.open()))
        assert len(rows) == 2
