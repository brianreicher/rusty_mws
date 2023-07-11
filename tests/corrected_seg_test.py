import unittest
import rusty_mws
from funlib.geometry import Coordinate
import numpy as np


class CorrectedSegTest(unittest.TestCase):
    def generate_fragments(self) -> None:

        completion: bool = rusty_mws.blockwise_generate_mutex_fragments_task(sample_name="test",
                                                                            affs_file="../data/raw_predictions.zarr",
                                                                            affs_dataset="pred_affs_latest",
                                                                            fragments_file="../data/raw_predictions.zarr",
                                                                            fragments_dataset="frag_seg",
                                                                            context=Coordinate(np.max(np.abs(rusty_mws.neighborhood), axis=0)),
                                                                            seeds_file="../data/raw_predictions.zarr",
                                                                            seeds_dataset="training_gt_rasters",
                                                                            training=True,)

        self.assertEqual(first=completion, second=True)

    # def test_correction(self) -> None:
    #     pass


if __name__ == "__main__":
    unittest.main()
