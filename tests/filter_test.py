import rusty_mws
from funlib.geometry import Coordinate
import numpy as np

import pytest


# stage data for testing
affs_data = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
fragments_data = np.array([1, 2, 1, 3])
filter_val = 0.5

def test_filter_below_threshold():
    # Test when all fragments should be filtered out
    rusty_mws.utils.filter_fragments(affs_data, fragments_data, filter_val)
    assert np.array_equal(fragments_data, np.array([]))

def test_filter_above_threshold():
    # Test when no fragments should be filtered out
    rusty_mws.utils.filter_fragments(affs_data, fragments_data, 0.0)
    assert np.array_equal(fragments_data, np.array([1, 2, 1, 3]))

def test_filter_partial():
    # Test when some fragments should be filtered out
    rusty_mws.utils.filter_fragments(affs_data, fragments_data, 0.3)
    assert np.array_equal(fragments_data, np.array([2, 3]))

# def test_generate_fragments() -> None:
#     task_completion: bool = rusty_mws.algo.blockwise_generate_mutex_fragments(
#         sample_name="test",
#         affs_file="../data/raw_predictions.zarr",
#         affs_dataset="pred_affs_latest",
#         fragments_file="../data/raw_predictions.zarr",
#         fragments_dataset="frag_seg",
#         context=Coordinate(np.max(np.abs(rusty_mws.neighborhood[:12]), axis=0)),
#     )

#     assert task_completion is True

# def test_generate_supervoxels() -> None:
#     task_completion: bool = rusty_mws.algo.blockwise_generate_supervoxel_edges(
#         sample_name="test",
#         affs_file="../data/raw_predictions.zarr",
#         affs_dataset="pred_affs_latest",
#         fragments_file="../data/raw_predictions.zarr",
#         fragments_dataset="frag_seg",
#         context=Coordinate(np.max(np.abs(rusty_mws.neighborhood[:12]), axis=0)),
#     )

#     assert task_completion is True

# def test_global_agglom() -> None:
#     task_completion: bool = rusty_mws.algo.global_mutex_agglomeration(
#         sample_name="test",
#         fragments_file="../data/raw_predictions.zarr",
#         fragments_dataset="frag_seg",
#     )

#     assert task_completion is True

# def test_global_agglom() -> None:
#     task_completion: bool = rusty_mws.algo.extract_segmentation(
#         fragments_file="../data/raw_predictions.zarr",
#         fragments_dataset="frag_seg",
#     )

#     assert task_completion is True

# def test_full_pred_segmentation_pipeline() -> None:
#     pp: rusty_mws.PostProcessor = rusty_mws.PostProcessor(
#         affs_file="../data/raw_predictions.zarr",
#         affs_dataset="pred_affs_latest",
#     )
#     task_completion: bool = pp.run_pred_segmentation_pipeline()

#     assert task_completion is True
