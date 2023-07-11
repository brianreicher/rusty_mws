import unittest
import rusty_mws
from funlib.geometry import Coordinate
import numpy as np


class CorrectedSegTest(unittest.TestCase):
    def generate_fragments(self) -> None:
        task_completion: bool = rusty_mws.blockwise_generate_mutex_fragments_task(
            sample_name="test",
            affs_file="../data/raw_predictions.zarr",
            affs_dataset="pred_affs_latest",
            fragments_file="../data/raw_predictions.zarr",
            fragments_dataset="frag_seg",
            context=Coordinate(np.max(np.abs(rusty_mws.neighborhood), axis=0)),
            seeds_file="../data/raw_predictions.zarr",
            seeds_dataset="training_gt_rasters",
            training=True,
        )

        self.assertEqual(first=task_completion, second=True)

    def test_skeleton_correction(self) -> None:
        task_completion: bool = rusty_mws.skel_correct_segmentation(
            raster_file="../data/raw_predictions.zarr",
            raster_dataset="training_gt_rasters",
            fragments_file="../data/raw_predictions.zarr",
            fragments_dataset="frag_seg",
            seg_file="../data/raw_predictions.zarr",
        )

        self.assertEqual(first=task_completion, second=True)

    def test_corrected_segmentation_full(self) -> None:
        task_completion: bool = rusty_mws.run_corrected_segmentation_pipleine(
            affs_file="../data/raw_predictions.zarr",
            affs_dataset="pred_affs_latest",
            fragments_file="../data/raw_predictions.zarr",
            fragments_dataset="frag_seg",
            seeds_file="../data/raw_predictions.zarr",
            seeds_dataset="training_gt_rasters",
            context=Coordinate(np.max(a=np.abs(rusty_mws.neighborhood), axis=0)),
        )
        self.assertEqual(first=task_completion, second=True)


if __name__ == "__main__":
    unittest.main()
