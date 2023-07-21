import logging
import time

import numpy as np
from scipy.ndimage import gaussian_filter, measurements

import daisy
import mwatershed as mws

from funlib.geometry import Coordinate, Roi
from funlib.persistence import open_ds, Array, graphs, prepare_ds
from funlib.segment.arrays import relabel
import pymongo

from ..utils import filter_fragments
from ..utils import neighborhood

logger: logging.Logger = logging.getLogger(__name__)


def blockwise_generate_mutex_fragments(
    sample_name: str,
    affs_file: str,
    affs_dataset: str,
    fragments_file: str,
    fragments_dataset: str,
    context: Coordinate,
    filter_val: float = 0.60,
    nworkers: int = 10,
    mask_file: str = None,
    mask_dataset: str = None,
    seeds_file: str = None,
    seeds_dataset: str = None,
    training: bool = False,
    n_chunk_write: int = 2,
    lr_bias_ratio: float = -0.175,
    adjacent_edge_bias: float = -0.4,  # bias towards merging
    neighborhood_length: int = 12,
    mongo_port: int = 27017,
    db_name: str = "seg",
) -> bool:
    """Generates MWS fragments and saves nodes & weights in a RAG.

    Args:
        sample_name (``str``):
            A string containing the sample name (run name of the experiment) to denote for the MongoDB collection_name

        affs_file (``str``):
            Path (relative or absolute) to the zarr file containing predicted affinities to generate fragments for.

        affs_dataset (``str``):
            The name of the affinities dataset in the affs_file to read from.

        fragments_file (``str``):
            Path (relative or absolute) to the zarr file to write fragments to.

        fragments_dataset (``str``):
            The name of the fragments dataset to read/write to in the fragments_file.

        context (``funlib.geometry.Coordinate``):
            A coordinate object (3-dimensional) denoting how much contextual space to grow for the total volume ROI.

        filter_val (``float``):
            The amount for which fragments will be filtered if their average falls below said value.

        nworkers (``integer``):
            Number of distributed workers to run the Daisy parallel task with.

        mask_file (``str``):
            Path (relative or absolute) to the zarr file containing an affinity mask.

        mask_dataset (``str``):
            The name of the mask dataset in the mask_file to read from.

        seeds_file (``str``):
            Path (relative or absolute) to the zarr file containing seeds.

        seeds_dataset (``str``):
            The name of the seeds dataset in the seeds file to read from.

        training (``bool``):
            Training flag to denote wether or not to storge fragment data in MongoDB. When ``false``, used for training-only runs on SLURM clusters.

        n_chunk_write (``integer``):
            Number of chunks to write for each Daisy block.

        lr_bias_ratio (``float``):
            Ratio at which to tweak the lr shift in offsets.

        adjacent_edge_bias (``float``):
            Weight base at which to bias adjacent edges.

        neighborhood_length (``integer``):
            Number of neighborhood offsets to use, default is 12.

        mongo_port (``integer``):
            Port number where a MongoDB server instance is listening.

        db_name (``string``):
            Name of the specified MongoDB database to use at the RAG.

        Returns:
            ``bool``:
                Returns ``true`` if all Daisy tasks complete successfully.
    """

    logger.info("Reading affs from %s", affs_file)
    logger.info(f"Experiment name: {sample_name}")

    affs: Array = open_ds(affs_file, affs_dataset, mode="r")
    chunk_shape: np.ndarray = np.array(affs.chunk_shape[affs.n_channel_dims :])
    num_voxels_in_block: np.ndarray = np.prod(chunk_shape * n_chunk_write)

    # new task params
    voxel_size: tuple = affs.voxel_size

    read_roi_voxels: Roi = Roi((0, 0, 0), chunk_shape * n_chunk_write).grow(
        context, context
    )

    write_roi_voxels: Roi = Roi((0, 0, 0), chunk_shape * n_chunk_write)

    total_roi_ds: Roi = affs.roi.grow(-context * voxel_size, -context * voxel_size)

    # Make total_roi_ds and even multiple of chunk_shape
    total_roi_ds: Roi = total_roi_ds.snap_to_grid(
        chunk_shape * voxel_size, mode="shrink"
    )

    # Add context to total_roi_ds for daisy
    total_roi_daisy: Roi = total_roi_ds.grow(context * voxel_size, context * voxel_size)

    read_roi: Roi = read_roi_voxels * voxel_size

    write_roi: Roi = write_roi_voxels * voxel_size

    logger.info("writing fragments to %s", fragments_file)

    fragments_prep: Array = prepare_ds(
        filename=fragments_file,
        ds_name=fragments_dataset,
        total_roi=total_roi_ds,
        voxel_size=voxel_size,
        dtype=np.uint64,
        delete=True,
    )

    fragments_ds: Array = open_ds(fragments_file, fragments_dataset, mode="r+")

    if mask_file is not None:
        logger.info("Reading mask from %s", mask_file)
        mask: Array = open_ds(mask_file, mask_dataset, mode="r")
    else:
        mask = None

    if seeds_file is not None:
        logger.info("Reading seeds from %s", seeds_file)
        seeds: Array = open_ds(seeds_file, seeds_dataset, mode="r")
    else:
        seeds = None

    if not training:
        # open RAG DB
        db_host: str = f"mongodb://localhost:{mongo_port}"

        mongo_drop = pymongo.MongoClient(db_host)[db_name]
        collection_names = mongo_drop.list_collection_names()

        for collection_name in collection_names:
            if sample_name in collection_name:
                logger.info(f"Dropping {collection_name}")
                mongo_drop[collection_name].drop()

        logger.info("Opening MongoDBGraphProvider...")
        rag_provider = graphs.MongoDbGraphProvider(
            db_name=db_name,
            host=db_host,
            mode="r+",
            directed=False,
            position_attribute=["center_z", "center_y", "center_x"],
            edges_collection=f"{sample_name}_edges",
            nodes_collection=f"{sample_name}_nodes",
            meta_collection=f"{sample_name}_meta",
        )

        logger.info("MongoDB Provider opened")

        # open block done DB
        mongo_client = pymongo.MongoClient(db_host)
        db = mongo_client[db_name]
        completed_collection = db[f"{sample_name}_fragment_blocks_extracted"]
    else:
        rag_provider = None
        completed_collection = None
    logger.info("Generating fragments . . .")

    # worker func
    def generate_mutex_fragments_worker(
        block: daisy.Block,
        affs=affs,
        seeds=seeds,
        context=context,
        fragments_out=fragments_ds,
        rag_provider=rag_provider,
        completed_collection=completed_collection,
        training=training,
    ) -> bool:
        start: float = time.time()

        logger.info("getting block")

        logger.info(f"got block {block}")
        logger.info("Performing seeded Rusty mutex watershed ...")

        logger.info("block read roi begin: %s", block.read_roi.get_begin())
        logger.info("block read roi shape: %s", block.read_roi.get_shape())
        logger.info("block write roi begin: %s", block.write_roi.get_begin())
        logger.info("block write roi shape: %s", block.write_roi.get_shape())

        offsets: list[list[int]] = neighborhood[:neighborhood_length]

        these_affs: Array = affs.intersect(block.read_roi)
        these_affs.materialize()

        if these_affs.dtype == np.uint8:
            logger.info("Assuming affinities are in [0,255]")
            max_affinity_value: float = 255.0
            these_affs.data = these_affs.data.astype(np.float64)
        else:
            max_affinity_value: float = 1.0

        these_affs.data /= max_affinity_value

        if these_affs.data.max() < 1e-3:
            return

        # extract fragments
        logger.info("Extracting frags")

        # add some random noise to affs (this is particularly necessary if your affs are
        #  stored as uint8 or similar)
        # If you have many affinities of the exact same value the order they are processed
        # in may be fifo, so you can get annoying streaks.
        logger.info("Making random noise")
        random_noise: float = np.random.randn(*these_affs.data.shape) * 0.001

        # add smoothed affs, to solve a similar issue to the random noise. We want to bias
        # towards processing the central regions of objects first.
        logger.info("Smoothing affs")
        smoothed_affs: np.ndarray = (
            gaussian_filter(these_affs.data, sigma=(0, *(Coordinate(context) / 3)))
            - 0.5
        ) * 0.01

        logger.info("Shifting affs")
        shift: np.ndarray = np.array(
            [
                adjacent_edge_bias
                if max(offset) <= 1
                else np.linalg.norm(offset) * lr_bias_ratio
                for offset in offsets
            ]
        ).reshape((-1, *((1,) * (len(these_affs.data.shape) - 1))))

        logger.info("Performing MWS")

        if seeds is not None:
            these_seeds: Array = seeds.intersect(block.read_roi)
            these_seeds.materialize()
            these_seeds.data = these_seeds.data.astype(np.uint64)
            fragments_data = mws.agglom(
                these_affs.data + shift + random_noise + smoothed_affs,
                offsets=offsets,
                seeds=these_seeds.data,
            )
        else:
            logger.info("Running unseeded agglom")
            fragments_data = mws.agglom(
                these_affs.data + shift + random_noise + smoothed_affs,
                offsets=offsets,
            )

        logger.info("Filtering fragments")
        if filter_val > 0.0:
            filter_fragments(these_affs, fragments_data, filter_val)

        fragments: Array = Array(fragments_data, these_affs.roi, these_affs.voxel_size)

        logger.info("Cropping Fragments")

        # crop fragments to write_roi
        fragments = fragments[block.write_roi]
        fragments.materialize()
        max_id: int = fragments.data.max()

        fragments.data, max_id = relabel(fragments.data)
        assert max_id < num_voxels_in_block
        # following only makes a difference if fragments were found
        if max_id == 0:
            return

        # ensure unique IDs
        id_bump: int = block.block_id[1] * num_voxels_in_block
        logger.info("bumping fragment IDs by %i", id_bump)
        fragments.data[fragments.data > 0] += id_bump
        fragment_ids = range(1 + id_bump, max_id + 1 + id_bump)

        # store fragments
        logger.info("Storing fragments")

        fragments_out[block.write_roi] = fragments

        logger.info("Storing in RagDB")
        if not training:
            # get fragment centers
            fragment_centers: dict = {
                fragment: block.write_roi.get_offset()
                + these_affs.voxel_size * Coordinate(center)
                for fragment, center in zip(
                    fragment_ids,
                    measurements.center_of_mass(
                        fragments.data, fragments.data, fragment_ids
                    ),
                )
                if not np.isnan(center[0])
            }

            rag = rag_provider[block.write_roi]
            rag.add_nodes_from(
                [
                    (node, {"center_z": c[0], "center_y": c[1], "center_x": c[2]})
                    for node, c in fragment_centers.items()
                ]
            )
            rag.write_nodes(block.write_roi)

            document: dict = {
                "block_id": block.block_id,
                "read_roi": (block.read_roi.get_begin(), block.read_roi.get_shape()),
                "write_roi": (block.write_roi.get_begin(), block.write_roi.get_shape()),
                "start": start,
                "duration": time.time() - start,
            }

            # add block to completed graph
            completed_collection.insert_one(document=document)

            logger.info(f"block information: {document}")
        logger.info(f"releasing block: {block}")
        logger.info(f"blocks completed:\n{block}")

        return True

    # create Daisy distributed task
    task: daisy.Task = daisy.Task(
        "MutexFragmentsTask",
        total_roi=total_roi_daisy,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=generate_mutex_fragments_worker,
        num_workers=nworkers,
        fit="shrink",
        read_write_conflict=False,
    )

    # run task blockwise
    ret: bool = daisy.run_blockwise(tasks=[task])
    return ret
