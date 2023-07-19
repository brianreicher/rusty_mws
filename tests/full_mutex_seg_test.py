import rusty_mws
from funlib.geometry import Coordinate
import numpy as np

import pytest


def test_generate_fragments() -> None:
    task_completion: bool = rusty_mws.algo.blockwise_generate_mutex_fragments(
        sample_name="test",
        affs_file="../data/raw_predictions.zarr",
        affs_dataset="pred_affs_latest",
        fragments_file="../data/raw_predictions.zarr",
        fragments_dataset="frag_seg",
        context=Coordinate(np.max(np.abs(rusty_mws.neighborhood[:12]), axis=0)),
    )

    assert task_completion is True

def test_generate_supervoxels() -> None:
    task_completion: bool = rusty_mws.algo.blockwise_generate_supervoxel_edges(
        sample_name="test",
        affs_file="../data/raw_predictions.zarr",
        affs_dataset="pred_affs_latest",
        fragments_file="../data/raw_predictions.zarr",
        fragments_dataset="frag_seg",
        context=Coordinate(np.max(np.abs(rusty_mws.neighborhood[:12]), axis=0)),
    )

    assert task_completion is True

def test_global_agglom() -> None:
    task_completion: bool = rusty_mws.algo.global_mutex_agglomeration(
        sample_name="test",
        fragments_file="../data/raw_predictions.zarr",
        fragments_dataset="frag_seg",
    )

    assert task_completion is True

def test_global_agglom() -> None:
    task_completion: bool = rusty_mws.algo.extract_segmentation(
        fragments_file="../data/raw_predictions.zarr",
        fragments_dataset="frag_seg",
    )

    assert task_completion is True

def test_full_pred_segmentation_pipeline() -> None:
    pp: rusty_mws.PostProcessor = rusty_mws.PostProcessor(
        affs_file="../data/raw_predictions.zarr",
        affs_dataset="pred_affs_latest",
    )
    task_completion: bool = pp.run_pred_segmentation_pipeline()

    assert task_completion is True
