import daisy
import logging
import numpy as np
import os
import time
from funlib.segment.arrays import replace_values
from funlib.persistence import open_ds, Array, prepare_ds
from funlib.geometry import Roi


logging.getLogger().setLevel(logging.INFO)


def extract_segmentation(
    fragments_file: str,
    fragments_dataset: str,
    nworkers: int = 20,
    merge_function: str = "mwatershed",
    n_chunk_write: int = 1,
) -> int:
    """Extracts and relabels fragments to segment IDs from the saved LUT.

    Args:
        fragments_file (``str``):
            Path (relative or absolute) to the zarr file where fragments are stored.

        fragments_dataset (``str``):
            The name of the fragments dataset to read from in the fragments file.

        nworkers (``integer``):
            Number of distributed workers to run the Daisy parallel task with.

        merge_function (``str``):
            Name of the segmentation algorithm used to denote in the MongoDB edge collection.

        n_chunk_write (``integer``):
            Number of chunks to write for each Daisy block.

    Returns:
        ``integer``:
            The number of unique segment IDs in the final segmentation.
    """

    lut_dir: str = os.path.join(fragments_file, "luts_full")

    fragments: Array = open_ds(fragments_file, fragments_dataset)

    voxel_size: tuple = fragments.voxel_size

    total_roi: Roi = fragments.roi
    chunk_shape: np.ndarray = np.array(fragments.chunk_shape)

    read_roi_voxels: Roi = Roi((0, 0, 0), chunk_shape * n_chunk_write)
    write_roi_voxels: Roi = read_roi_voxels

    read_roi: Roi = read_roi_voxels * voxel_size
    write_roi: Roi = write_roi_voxels * voxel_size

    logging.info("Preparing segmentation dataset...")

    seg_name: str = f"pred_seg"

    start: float = time.time()

    segmentation: Array = prepare_ds(
        fragments_file,
        seg_name,
        fragments.roi,
        voxel_size=voxel_size,
        dtype=np.uint64,
        write_roi=write_roi,
        delete=True,
    )

    lut_filename: str = f"seg_{merge_function}"

    lut: str = os.path.join(lut_dir, lut_filename + ".npz")

    assert os.path.exists(lut), f"{lut} does not exist"

    logging.info("Reading fragment-segment LUT...")

    lut: np.ndarray = np.load(lut)["fragment_segment_lut"]

    logging.info(f"Found {len(lut[0])} fragments in LUT")

    num_segments: int = len(np.unique(lut[1]))
    logging.info(f"Relabelling fragments to {num_segments} segments")

    task: daisy.Task = daisy.Task(
        "ExtractSegmentationTask",
        total_roi,
        read_roi,
        write_roi,
        lambda b: segment_in_block(b, segmentation, fragments, lut),
        fit="shrink",
        num_workers=nworkers,
    )

    done: bool = daisy.run_blockwise(tasks=[task])

    if not done:
        raise RuntimeError(
            "Extraction of segmentation from LUT failed for (at least) one block"
        )

    logging.info(f"Took {time.time() - start} seconds to extract segmentation from LUT")
    return num_segments


def segment_in_block(block: daisy.Block, segmentation, fragments, lut) -> bool:
    logging.info("Copying fragments to memory...")

    # load fragments
    fragments: np.ndarray = fragments.to_ndarray(block.read_roi)

    # replace values, write to empty array
    relabelled: np.ndarray = np.zeros_like(fragments)
    old_vals: np.ndarray = np.array(lut[0], dtype=np.uint64)
    new_vals: np.ndarray = np.array(lut[1], dtype=np.uint64)
    assert old_vals.dtype == new_vals.dtype == fragments.dtype

    logging.info("Relabelling . . .")
    relabelled: np.ndarray = replace_values(fragments, old_vals, new_vals, out_array=relabelled)

    segmentation[block.write_roi] = relabelled
    return True
