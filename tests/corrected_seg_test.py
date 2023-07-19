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
        seeds_file="../data/raw_predictions.zarr",
        seeds_dataset="training_gt_rasters",
        training=True,
    )

    assert task_completion is True

def test_skeleton_correction() -> None:
    task_completion: bool = rusty_mws.algo.skel_correct_segmentation(
        seeds_file="../data/raw_predictions.zarr",
        seeds_dataset="training_gt_rasters",
        fragments_file="../data/raw_predictions.zarr",
        fragments_dataset="frag_seg",
        seg_file="../data/raw_predictions.zarr",
    )

    assert task_completion is True

def test_corrected_segmentation_full() -> None:
    pp = rusty_mws.PostProcessor(
        affs_file="../data/raw_predictions.zarr",
        affs_dataset="pred_affs_latest",
        seeds_file="../data/raw_predictions.zarr",
        seeds_dataset="training_gt_rasters",
    )
    task_completion: bool = pp.run_corrected_segmentation_pipeline()

    assert task_completion is True
