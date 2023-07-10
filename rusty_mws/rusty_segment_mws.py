import logging
import time
import os
from funlib.geometry import Coordinate
from .mutex_fragments_worker import *
from .supervoxels_worker import *
from .global_mutex import *
from .extract_seg_from_luts import *
from .skeleton_correct import *

logger: logging.Logger = logging.getLogger(name=__name__)


def get_corrected_segmentation(
    affs_file: str,
    affs_dataset,
    fragments_file: str,
    fragments_dataset,
    seeds_file: str,
    seeds_dataset: str,
    context: Coordinate,
    filter_fragments: float,
    seg_file: str = "./raw_predictions.zarr",
    seeded: bool = True,
    cutout: bool = False,
    # sample_name: str = "htem39454661040933637",
    sample_name=None,
) -> bool:
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
        success = success & blockwise_generate_mutex_fragments_task(
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
        success = success & blockwise_generate_mutex_fragments_task(
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
    )

    return success


def get_pred_segmentation(
    affs_file: str,
    affs_dataset,
    fragments_file: str,
    fragments_dataset,
    context: list,
    filter_fragments: float,
    adj_bias: float,
    lr_bias: float,
    generate_frags_and_edges: bool = False,
    # sample_name: str = "htem39454661040933637",
    sample_name=None,
) -> bool:
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
        success = success & blockwise_generate_mutex_fragments_task(
            sample_name,
            affs_file,
            affs_dataset,
            fragments_file,
            fragments_dataset,
            context,
            filter_fragments,
            training=False,
        )
        success = success & blockwise_generate_super_voxel_edges_task(
            sample_name,
            affs_file,
            affs_dataset,
            fragments_file,
            fragments_dataset,
            context,
        )

    success = success & global_mutex_watershed_on_super_voxels(
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
    global_mutex_watershed_on_super_voxels(
        fragments_file,
        fragments_dataset,
        sample_name,
        adj_bias=adj_bias,
        lr_bias=lr_bias,
    )
    extract_segmentation(fragments_file, fragments_dataset, sample_name, num_workers=20)
    return True
