"""Tests for preserving feature and target identity when loading CSV data."""

import numpy as np
import pytest

from src.experiment.common import load_csv_dataset


@pytest.mark.parametrize("header,target", [
    ("x1,x1,y", "y"),
    ("x1,y,y", "y"),
    ("x1,response,response", "response"),
])
def test_duplicate_csv_columns_are_rejected(tmp_path, header, target):
    """Duplicate feature or target names must not silently discard values."""
    path = tmp_path / "duplicates.csv"
    path.write_text(f"{header}\n1,10,3\n2,20,5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_csv_dataset(path, target_column=target)


@pytest.mark.parametrize("header,row,target", [
    ("x1,y", "1,2,999", "y"),
    ("x1,y", "1", "y"),
    ("response,x1,x2", "1,2", "response"),
])
def test_csv_rows_must_match_header_width(tmp_path, header, row, target):
    """Extra or missing fields must not silently alter the fitting dataset."""
    path = tmp_path / "invalid_width.csv"
    path.write_text(f"{header}\n{row}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="same number of fields"):
        load_csv_dataset(path, target_column=target)


def test_unique_csv_columns_preserve_values_and_feature_order(tmp_path):
    """A target between feature columns must leave their order intact."""
    path = tmp_path / "unique.csv"
    path.write_text("x2,response,x1\n1,10,3\n2,20,5\n", encoding="utf-8")
    x_values, y_values, columns = load_csv_dataset(path, target_column="response")
    assert columns == ["x2", "x1"]
    np.testing.assert_array_equal(x_values, [[1, 3], [2, 5]])
    np.testing.assert_array_equal(y_values, [10, 20])
