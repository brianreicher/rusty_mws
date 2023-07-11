import unittest
import rusty_mws
from funlib.geometry import Coordinate
import numpy as np


class CorrectedSegTest(unittest.TestCase):
    def generate_fragments(self):

        completion: bool = rusty_mws.mutex_fragments_worker.blockwise_generate_mutex_fragments_task(sample_name="test",
                                                                                                    affs_file="../data/raw_predictions.zarr",
                                                                                                    affs_dataset="pred_affs_latest",
                                                                                                    fragments_file="../data/raw_predictions.zarr",
                                                                                                    fragments_dataset="frag_seg",
                                                                                                    context=Coordinate(np.max(np.abs(rusty_mws.neighborhood), axis=0)),
)

        self.assertEqual(first=True, second=True) # TODO

    def test_correction(self):
        pass


if __name__ == "__main__":
    unittest.main()
