from tqdm import tqdm
import numpy as np
import daisy
from skimage.morphology import ball, erosion, dilation
import logging
from funlib.persistence import Array, open_ds, prepare_ds
from funlib.geometry import Coordinate, Roi


logger: logging.Logger = logging.getLogger(__name__)


def skel_correct_segmentation(
    seeds_file: str,
    seeds_dataset: str,
    fragments_file: str,
    fragments_dataset: str,
    seg_file: str,
    seg_dataset: str,
    nworkers: int = 25,
    erode_iterations: int = 0,
    erode_footprint: np.ndarray = ball(radius=5),
    alternate_dilate: bool = True,
    dilate_footprint: np.ndarray = ball(radius=5),
    n_chunk_write: int = 1,
) -> bool:
    """Corrects inintial fragments using pre-defined skeletons to create an agglomerated full segmentation.

    Args:

    seeds_file (``str``):
        Path (relative or absolute) to the zarr file containing fragments.

    seeds_dataset (``str``):
        The name of the fragments dataset to read from.


    fragments_file (``str``):
        Path (relative or absolute) to the zarr file containing fragments.

    fragments_dataset (``str``):
        The name of the fragments dataset to read from.

    seg_file (``str``):
        Path (relative or absolute) to the zarr file to write fragments to.

    seg_dataset (``str``):
        The name of the segmentation dataset to write to.

    nworkers (``integer``):
            Number of distributed workers to run the Daisy parallel task with.

    erode_iterations (``integer``):
        Number of iterations to erode/dialate agglomerated fragments.

    erode_footprint (``np.ndarray``):
        Numpy array denoting a ball of a given radius to erode segments by.

    alternate_dilate (``bool``):
        Flag that will allow for alterate erosions/dialations during segmentation.

    dialate_footprint (``np.ndarray``):
        Numpy array denoting a ball of a given radius to dialate segments by.

    n_chunk_write (``integer``):
            Number of chunks to write for each Daisy block.

    Returns:
        ``bool``:
            Returns ``true`` if all Daisy tasks complete successfully.
    """

    frags: Array = open_ds(filename=fragments_file, ds_name=fragments_dataset)
    raster_ds: Array = open_ds(filename=seeds_file, ds_name=seeds_dataset)
    chunk_shape: tuple = frags.chunk_shape[frags.n_channel_dims :]

    # task params
    voxel_size: Coordinate = frags.voxel_size
    read_roi_voxels: Roi = Roi(
        offset=(0, 0, 0), shape=chunk_shape * n_chunk_write
    )  # TODO: may want to add context here
    write_roi_voxels: Roi = Roi(offset=(0, 0, 0), shape=chunk_shape * n_chunk_write)
    total_roi: Roi = frags.roi
    dtype = frags.dtype
    read_roi: Roi = read_roi_voxels * voxel_size
    write_roi: Roi = write_roi_voxels * voxel_size

    # setup output zarr
    seg_ds: Array = prepare_ds(
        filename=seg_file,
        ds_name=seg_dataset,
        total_roi=total_roi,
        voxel_size=voxel_size,
        dtype=dtype,
        delete=True,
    )

    # setup labels_mask zarr
    ds: Array = prepare_ds(
        seg_file,
        "pred_labels_mask",
        total_roi=total_roi,
        voxel_size=voxel_size,
        dtype=np.uint8,
        delete=True,
    )

    # setup unlabelled_mask zarr
    ds: Array = prepare_ds(
        seg_file,
        ds_name="pred_unlabelled_mask",
        total_roi=total_roi,
        voxel_size=voxel_size,
        dtype=np.uint8,
        delete=True,
    )

    seg_ds: Array = open_ds(filename=seg_file, ds_name=seg_dataset, mode="r+")

    def skel_correct_worker(
        block: daisy.Block, seg_ds=seg_ds, raster_ds=raster_ds
    ) -> bool:
        raster_array: np.ndarray = raster_ds.to_ndarray(block.read_roi)
        frag_array: np.ndarray = frags.to_ndarray(block.read_roi)
        assert raster_array.shape == frag_array.shape

        seg_array: np.ndarray = np.zeros_like(frag_array)

        for frag_id in tqdm(np.unique(frag_array)):
            if frag_id == 0:
                continue
            seg_ids: list = list(np.unique(raster_array[frag_array == frag_id]))

            if seg_ids[0] == 0:
                seg_ids.pop(0)
            if len(seg_ids) == 1:
                seg_array[frag_array == frag_id] = seg_ids[0]

        if erode_iterations > 0:
            for _ in range(erode_iterations):
                seg_array = erosion(seg_array, erode_footprint)
                if alternate_dilate:
                    seg_array = dilation(seg_array, dilate_footprint)

        logger.info("writing segmentation to disk")

        seg_ds[block.write_roi] = seg_array

        # Now make labels mask
        labels_mask: np.ndarray = np.ones_like(seg_array).astype(np.uint8)
        labels_mask_ds: Array = open_ds(seg_file, "pred_labels_mask", mode="a")
        labels_mask_ds[block.write_roi] = labels_mask

        # Now make the unlabelled mask
        unlabelled_mask: np.ndarray = (seg_array > 0).astype(np.uint8)
        unlabelled_mask_ds: Array = open_ds(seg_file, "pred_unlabelled_mask", mode="a")
        unlabelled_mask_ds[block.write_roi] = unlabelled_mask
        return True

    # create task
    task: daisy.Task = daisy.Task(
        task_id="UpdateSegTask",
        total_roi=total_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=skel_correct_worker,
        num_workers=nworkers,
    )

    # run task
    ret: bool = daisy.run_blockwise([task])
    return ret
