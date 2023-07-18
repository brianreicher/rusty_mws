import unittest
import rusty_mws
from funlib.geometry import Coordinate
import numpy as np


class CorrectedSegTest(unittest.TestCase):
    def test_generate_fragments(self) -> None:
        task_completion: bool = rusty_mws.blockwise_generate_mutex_fragments_task(
            sample_name="test",
            affs_file="../data/raw_predictions.zarr",
            affs_dataset="pred_affs_latest",
            fragments_file="../data/raw_predictions.zarr",
            fragments_dataset="frag_seg",
            context=Coordinate(np.max(np.abs(rusty_mws.neighborhood), axis=0)),
        )

        self.assertEqual(first=task_completion, second=True)

    def test_generate_supervoxels(self) -> None:
        task_completion: bool = rusty_mws.blockwise_generate_supervoxel_edges(
            sample_name="test",
            affs_file="../data/raw_predictions.zarr",
            affs_dataset="pred_affs_latest",
            fragments_file="../data/raw_predictions.zarr",
            fragments_dataset="frag_seg",
            context=Coordinate(np.max(np.abs(rusty_mws.neighborhood), axis=0)),
        )

        self.assertEqual(first=task_completion, second=True)

    def test_global_agglom(self) -> None:
        task_completion: bool = rusty_mws.global_mutex_agglomeration(
            sample_name="test",
            fragments_file="../data/raw_predictions.zarr",
            fragments_dataset="frag_seg",
        )

        self.assertEqual(first=task_completion, second=True)

    def test_global_agglom(self) -> None:
        task_completion: bool = rusty_mws.extract_segmentation(
            fragments_file="../data/raw_predictions.zarr",
            fragments_dataset="frag_seg",
        )

        self.assertEqual(first=task_completion, second=True)

    def test_full_pred_segmentation_pipeline(self) -> None:
        task_completion: bool = rusty_mws.run_pred_segmentation_pipeline(
            affs_file="../data/raw_predictions.zarr",
            affs_dataset="pred_affs_latest",
            fragments_file="../data/raw_predictions.zarr",
            fragments_dataset="frag_seg",
            context=Coordinate(np.max(a=np.abs(rusty_mws.neighborhood), axis=0)),
        )
        self.assertEqual(first=task_completion, second=True)


if __name__ == "__main__":
    unittest.main()
