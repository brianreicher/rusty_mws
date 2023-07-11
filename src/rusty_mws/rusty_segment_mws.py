import logging
import time
import os
from funlib.geometry import Coordinate
from .generate_mutex_fragments import *
from .generate_supervoxel_edges import *
from .global_mutex_agglom import *
from .extract_seg_from_luts import *
from .skeleton_correct import *


logger: logging.Logger = logging.getLogger(name=__name__)


def get_corrected_segmentation(
    affs_file: str,
    affs_dataset: str,
    fragments_file: str,
    fragments_dataset,
    seeds_file: str,
    seeds_dataset: str,
    context: Coordinate,
    filter_fragments: float,
    seg_file: str = "./raw_predictions.zarr",
    seg_dataset: str = "pred_seg",
    seeded: bool = True,
    sample_name=None,
) -> bool:
    """Full skeleton-corrected Mutex segmentation from affinities.

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

        context (``daisy.Coordinate``):
            A coordinate object (3-dimensional) denoting how much contextual space to grow for the total volume ROI.

        filter_val (``float``):
            The amount for which fragments will be filtered if their average falls below said value.

        seg_file (``str``):
            Path (relative or absolute) to the zarr file to write fragments to.

        seg_dataset (``str``):
            The name of the segmentation dataset to write to.

        seeded (``bool``):
            Flag to determine whether or not to create seeded Mutex fragments.

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
    if seeded:
        success = success & blockwise_generate_mutex_fragments(
            sample_name=sample_name,
            affs_file=affs_file,
            affs_dataset=affs_dataset,
            fragments_file=fragments_file,
            fragments_dataset=fragments_dataset,
            context=context,
            filter_val=filter_fragments,
            seeds_file=seeds_file,
            seeds_dataset=seeds_dataset,
            training=True,
        )
    else:
        success = success & blockwise_generate_mutex_fragments(
            sample_name=sample_name,
            affs_file=affs_file,
            affs_dataset=affs_dataset,
            fragments_file=fragments_file,
            fragments_dataset=fragments_dataset,
            context=context,
            filter_val=filter_fragments,
            seeds_file=None,
            seeds_dataset=None,
        )

    success = success & skel_correct_segmentation(
        raster_file=seeds_file,
        raster_name=seeds_dataset,
        frag_file=fragments_file,
        frag_name=fragments_dataset,
        seg_file=seg_file,
        seg_dataset=seg_dataset,
    )

    return success


def get_pred_segmentation(
    affs_file: str,
    affs_dataset: str,
    fragments_file: str,
    fragments_dataset,
    context: list,
    filter_fragments: float,
    adj_bias: float,
    lr_bias: float,
    generate_frags_and_edges: bool = False,
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


def optimize_pred_segmentation(
    adj_bias: float,
    lr_bias: float,
    sample_name: str = "htem4413041148969302336",
    fragments_file: str = "./validation.zarr",
    fragments_dataset: str = "frag_seg",
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
