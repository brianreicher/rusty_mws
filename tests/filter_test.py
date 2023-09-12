import rusty_mws
from funlib.geometry import Coordinate
import numpy as np

import pytest


def test_filter_partial():
    # Test when all fragments should be filtered out
    affs_data = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    fragments_data = np.array([1, 2, 1, 3])
    filter_val = 0.5
    rusty_mws.utils.filter_fragments(affs_data, fragments_data, filter_val)
    assert np.array_equal(fragments_data, np.array([0, 0, 0, 3]))


def test_filter_above_threshold():
    # Test when no fragments should be filtered out
    affs_data = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    fragments_data = np.array([1, 2, 1, 3])
    filter_val = 0.0
    rusty_mws.utils.filter_fragments(affs_data, fragments_data, filter_val)
    assert np.array_equal(fragments_data, np.array([1, 2, 1, 3]))
