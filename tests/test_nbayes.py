import os
import pytest
from classifier import nbayes

HERE = os.path.dirname(__file__)

@pytest.mark.parametrize("suffix,expected", [
    ("a", 21.1697574893),
    ("b", 10.3438635728),
])
def test_nbayes_returns_expected_hF(suffix, expected):
    """
    Verifies that nbayes(False, True, …) returns the known hF for our two small datasets.
    """
    train_file = os.path.join(HERE, "datasets", "train",   f"train_{suffix}.arff")
    test_file  = os.path.join(HERE, "datasets", "test",    f"test_{suffix}.arff")
    result = nbayes(False, True, train_file, test_file, False)
    # allow for tiny FP rounding diffs
    assert result == pytest.approx(expected, rel=1e-3)
