import daisy
import logging
import numpy as np
import os
import time
from funlib.segment.arrays import replace_values


logging.getLogger().setLevel(logging.INFO)

def extract_segmentation(
        fragments_file,
        fragments_dataset,
        sample_name,
        num_workers:int=20,
        merge_function:str="mwatershed",
        n_chunk_write:int=1):
        
    lut_dir = os.path.join(fragments_file,'luts_full')

    fragments = daisy.open_ds(fragments_file, fragments_dataset)

    # block_size = fragments.roi.shape
    voxel_size = fragments.voxel_size

    total_roi = fragments.roi
    chunk_shape = np.array(fragments.chunk_shape)

    read_roi_voxels=daisy.Roi((0, 0, 0), chunk_shape*n_chunk_write)
    write_roi_voxels=read_roi_voxels

    read_roi = read_roi_voxels * voxel_size
    write_roi = write_roi_voxels * voxel_size

    logging.info("Preparing segmentation dataset...")

    seg_name = f"pred_seg"

    start = time.time()

    segmentation = daisy.prepare_ds(
        fragments_file,
        seg_name,
        fragments.roi,
        voxel_size=voxel_size,
        dtype=np.uint64,
        write_roi=write_roi,
        delete=True)

    lut_filename = f'seg_{merge_function}'

    lut = os.path.join(
            lut_dir,
            lut_filename + '.npz')

    assert os.path.exists(lut), f"{lut} does not exist"

    logging.info("Reading fragment-segment LUT...")

    lut = np.load(lut)['fragment_segment_lut']

    logging.info(f"Found {len(lut[0])} fragments in LUT")

    num_segments = len(np.unique(lut[1]))
    logging.info(f"Relabelling fragments to {num_segments} segments")

    task = daisy.Task(
        'ExtractSegmentationTask',
        total_roi,
        read_roi,
        write_roi,
        lambda b: segment_in_block(
            b,
            segmentation,
            fragments,
            lut),
        fit='shrink',
        num_workers=num_workers)

    done: bool = daisy.run_blockwise(tasks=[task])

    if not done:
        raise RuntimeError("Extraction of segmentation from LUT failed for (at least) one block")

    logging.info(f"Took {time.time() - start} seconds to extract segmentation from LUT")
    return num_segments

def segment_in_block(
        block,
        segmentation,
        fragments,
        lut):

    logging.info("Copying fragments to memory...")

    # load fragments
    fragments = fragments.to_ndarray(block.read_roi)

    # replace values, write to empty array
    relabelled = np.zeros_like(fragments)
    old_vals: np.ndarray = np.array(lut[0], dtype=np.uint64)
    new_vals: np.ndarray = np.array(lut[1], dtype=np.uint64)
    assert old_vals.dtype == new_vals.dtype == fragments.dtype

    logging.info("Relabelling . . .")
    relabelled = replace_values(fragments, old_vals, new_vals, out_array=relabelled)

    segmentation[block.write_roi] = relabelled
    return True
