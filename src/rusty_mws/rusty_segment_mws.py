import logging
import time
import os
from funlib.geometry import Coordinate
from algo.generate_mutex_fragments import *
from algo.generate_supervoxel_edges import *
from algo.global_mutex_agglom import *
from algo.extract_seg_from_luts import *
from algo.skeleton_correct import *


logger: logging.Logger = logging.getLogger(name=__name__)


def run_corrected_segmentation_pipeline(
    affs_file: str,
    affs_dataset: str,
    fragments_file: str,
    fragments_dataset,
    seeds_file: str,
    seeds_dataset: str,
    seg_file: str,
    seg_dataset: str,
    context: Coordinate,
    sample_name=None,
    mask_file: str = None,
    mask_dataset: str = None,
    filter_val: float = 0.5,
    nworkers_frags: int = 10,
    n_chunk_write_frags: int = 2,
    lr_bias_ratio: float = -0.175,
    adjacent_edge_bias: float = -0.4,
    neighborhood_length: int = 12,
    mongo_port: int = 27017,
    db_name: str = "seg",
    seeded: bool = True,
    nworkers_correct: int = 25,
    n_chunk_write_correct: int = 1,
    erode_iterations: int = 0,
    erode_footprint: np.ndarray = ball(radius=5),
    alternate_dilate: bool = True,
    dilate_footprint: np.ndarray = ball(radius=5),
) -> bool:
    """Full skeleton-corrected MWS segmentation from affinities.

    Args:
        affs_file (``str``):
            Path (relative or absolute) to the zarr file containing predicted affinities to generate fragments for.

        affs_dataset (``str``):
            The name of the affinities dataset in the affs_file to read from.

        fragments_file (``str``):
            Path (relative or absolute) to the zarr file to write fragments to.

        fragments_dataset (``str``):
            The name of the fragments dataset to read/write to in the fragments_file.

        seeds_file (``str``):
            Path (relative or absolute) to the zarr file containing seeds.

        seeds_dataset (``str``):
            The name of the seeds dataset in the seeds file to read from.
        
        seg_file (``str``):
            Path (relative or absolute) to the zarr file to write fragments to.

        seg_dataset (``str``):
            The name of the segmentation dataset to write to.

        context (``daisy.Coordinate``):
            A coordinate object (3-dimensional) denoting how much contextual space to grow for the total volume ROI.
        
        sample_name (``str``):
            A string containing the sample name (run name of the experiment) to denote for the MongoDB collection_name.

        filter_val (``float``):
            The amount for which fragments will be filtered if their average falls below said value.

        nworkers_frags (``integer``):
            Number of distributed workers to run the Daisy parallel fragment task with.
        
        n_chunk_write_frags (``integer``):
            Number of chunks to write for each Daisy block in the fragment task.

        lr_bias_ratio (``float``):
            Ratio at which to tweak the lr shift in offsets.

        adjacent_edge_bias (``float``):
            Weight base at which to bias adjacent edges.

        neighborhood_length (``integer``):
            Number of neighborhood offsets to use, default is 8.
        
        mongo_port (``integer``):
            Port number where a MongoDB server instance is listening.
        
        db_name (``string``):
            Name of the specified MongoDB database to use at the RAG.

        seeded (``bool``):
            Flag to determine whether or not to create seeded Mutex fragments.

        nworkers_correct (``integer``):
            Number of distributed workers to run the Daisy parallel skeleton correction task with.
        
        n_chunk_write_correct (``integer``):
            Number of chunks to write for each Daisy block in the skeleton correction task.

        erode_iterations (``integer``):
            Number of iterations to erode/dialate agglomerated fragments.

        erode_footprint (``np.ndarray``):
            Numpy array denoting a ball of a given radius to erode segments by.

        alternate_dilate (``bool``):
            Flag that will allow for alterate erosions/dialations during segmentation.

        dialate_footprint (``np.ndarray``):
            Numpy array denoting a ball of a given radius to dialate segments by.

    Returns:
        ``bool``:
            Denotes whether or not the segmentation is completed successfully.
    """

    if sample_name is None:
        sample_name: str = "htem" + str(
            hash(
                f"FROM{os.path.join(affs_file, affs_dataset)}TO{os.path.join(fragments_file, fragments_dataset)}AT{time.strftime('%Y%m%d-%H%M%S')}".replace(
                    ".", "-"
                ).replace(
                    "/", "-"
                )
            )
        )

    success: bool = True
    if seeded:
        success = success & blockwise_generate_mutex_fragments(
            sample_name=sample_name,
            affs_file=affs_file,
            affs_dataset=affs_dataset,
            fragments_file=fragments_file,
            fragments_dataset=fragments_dataset,
            context=context,
            filter_val=filter_val,
            seeds_file=seeds_file,
            seeds_dataset=seeds_dataset,
            mask_file=mask_file,
            mask_dataset=mask_dataset,
            training=True,
            nworkers=nworkers_frags,
            n_chunk_write=n_chunk_write_frags,
            lr_bias_ratio=lr_bias_ratio,
            adjacent_edge_bias=adjacent_edge_bias,
            neighborhood_length=neighborhood_length,
            mongo_port=mongo_port,
            db_name=db_name,
        )
    else:
        success = success & blockwise_generate_mutex_fragments(
            sample_name=sample_name,
            affs_file=affs_file,
            affs_dataset=affs_dataset,
            fragments_file=fragments_file,
            fragments_dataset=fragments_dataset,
            context=context,
            filter_val=filter_val,
            seeds_file=None,
            seeds_dataset=None,
            training=True,
            mask_file=mask_file,
            mask_dataset=mask_dataset,
            training=True,
            nworkers=nworkers_frags,
            n_chunk_write=n_chunk_write_frags,
            lr_bias_ratio=lr_bias_ratio,
            adjacent_edge_bias=adjacent_edge_bias,
            neighborhood_length=neighborhood_length,
            mongo_port=mongo_port,
            db_name=db_name,
        )

    success = success & skel_correct_segmentation(
        seeds_file=seeds_file,
        seeds_dataset=seeds_dataset,
        frag_file=fragments_file,
        frag_name=fragments_dataset,
        seg_file=seg_file,
        seg_dataset=seg_dataset,
        nworkers=nworkers_correct,
        erode_iterations=erode_iterations,
        erode_footprint=erode_footprint,
        alternate_dilate=alternate_dilate,
        dilate_footprint=dilate_footprint,
        n_chunk_write=n_chunk_write_correct,
    )

    return success


def run_pred_segmentation_pipeline(
    affs_file: str,
    affs_dataset: str,
    fragments_file: str,
    fragments_dataset,
    context: list,
    filter_fragments: float = 0.5,
    adj_bias: float = -0.1,
    lr_bias: float = -1.5,
    generate_frags_and_edges: bool = True,
    sample_name=None,
) -> bool:
    """Full Mutex Watershed segmentation and agglomeration, using a MongoDB graph.

    Args:
        affs_file (``str``):
            Path (relative or absolute) to the zarr file containing predicted affinities to generate fragments for.

        affs_dataset (``str``):
            The name of the affinities dataset in the affs_file to read from.

        fragments_file (``str``):
            Path (relative or absolute) to the zarr file to write fragments to.

        fragments_dataset (``str``):
            The name of the fragments dataset to read/write to in the fragments_file.

        context (``daisy.Coordinate``):
            A coordinate object (3-dimensional) denoting how much contextual space to grow for the total volume ROI.

        filter_val (``float``):
            The amount for which fragments will be filtered if their average falls below said value.

        adj_bias (``float``):
            Amount to bias adjacent pixel weights when computing segmentation from the stored graph.

        lr_bias (``float``):
            Amount to bias long-range pixel weights when computing segmentation from the stored graph.

        generate_frags_and_edges (``bool``):
            Flag whether or not to generate fragments and edges or solely perform agglomeration.

        sample_name (``str``):
            A string containing the sample name (run name of the experiment) to denote for the MongoDB collection_name.

    Returns:
        ``bool``:
            Denotes whether or not the segmentation is completed successfully.
    """

    if sample_name is None:
        sample_name: str = "htem" + str(
            hash(
                f"FROM{os.path.join(affs_file, affs_dataset)}TO{os.path.join(fragments_file, fragments_dataset)}AT{time.strftime('%Y%m%d-%H%M%S')}".replace(
                    ".", "-"
                ).replace(
                    "/", "-"
                )
            )
        )

    success: bool = True

    if generate_frags_and_edges:
        success = success & blockwise_generate_mutex_fragments(
            sample_name,
            affs_file,
            affs_dataset,
            fragments_file,
            fragments_dataset,
            context,
            filter_fragments,
            training=False,
        )
        success = success & blockwise_generate_supervoxel_edges(
            sample_name,
            affs_file,
            affs_dataset,
            fragments_file,
            fragments_dataset,
            context,
        )

    success = success & global_mutex_agglomeration(
        fragments_file,
        fragments_dataset,
        sample_name=sample_name,
        adj_bias=adj_bias,
        lr_bias=lr_bias,
    )

    success = success & extract_segmentation(
        fragments_file,
        fragments_dataset,
        sample_name=sample_name,
        num_workers=75,
    )

    return True


def optimize_pred_segmentation( # TODO: implement genetic optimization
    adj_bias: float,
    lr_bias: float,
    sample_name: str,
    fragments_file: str,
    fragments_dataset: str,
) -> bool:
    """Soley global agglomeration and segment extraction via Mutex Watershed - used to optimize weights during the global agglomeration step.

    Args:
        adj_bias (``float``):
            Amount to bias adjacent pixel weights when computing segmentation from the stored graph.

        lr_bias (``float``):
            Amount to bias long-range pixel weights when computing segmentation from the stored graph.

        sample_name (``str``):
            A string containing the sample name (run name of the experiment) to denote for the MongoDB collection_name.

        fragments_file (``str``):
            Path (relative or absolute) to the zarr file to read fragments from.

        fragments_dataset (``str``):
            The name of the fragments dataset to read from in the fragments_file.

    """
    global_mutex_agglomeration(
        fragments_file,
        fragments_dataset,
        sample_name,
        adj_bias=adj_bias,
        lr_bias=lr_bias,
    )
    extract_segmentation(fragments_file, fragments_dataset, sample_name, num_workers=20)
    return True
