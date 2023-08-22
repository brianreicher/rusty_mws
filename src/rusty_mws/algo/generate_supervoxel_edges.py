import itertools
import logging
import time

import numpy as np
from scipy.ndimage import measurements

import pymongo
import daisy
from funlib.geometry import Coordinate, Roi
from funlib.persistence import open_ds, Array, graphs
from ..utils import neighborhood


logger: logging.Logger = logging.getLogger(__name__)


def blockwise_generate_supervoxel_edges(
    sample_name: str,
    affs_file: str,
    affs_dataset,
    fragments_file: str,
    fragments_dataset: str,
    context: Coordinate,
    nworkers: int = 20,
    merge_function: str = "mwatershed",
    lr_bias_ratio: float = -0.175,
    neighborhood_length: int = 12,
    mongo_port: int = 27017,
    db_name: str = "seg",
    use_mongo: bool = True,
) -> bool:
    """Generates supervoxel edges and stores (u, v, adj, lr) weights in a RAG.

    Args:
        sample_name (``str``):
            A string containing the sample name (run name of the experiment) to denote for the MongoDB collection_name

        affs_file (``str``):
            Path (relative or absolute) to the zarr file containing predicted affinities to generate fragments for.

        affs_dataset (``str``):
            The name of the affinities dataset in the affs_file to read from.

        fragments_file (``str``):
            Path (relative or absolute) to the zarr file containing fragments.

        fragments_dataset (``str``):
            The name of the fragments dataset to read from.

        context (``daisy.Coordinate``):
            A coordinate object (3-dimensional) denoting how much contextual space to grow for the total volume ROI.

        nworkers (``integer``):
            Number of distributed workers to run the Daisy parallel task with.

        merge_function (``str``):
            Name of the segmentation algorithm used to denote in the MongoDB edge collection.

        lr_bias_ratio (``float``):
            Ratio at which to tweak the lr shift in offsets.

        neighborhood_length (``integer``):
            Number of neighborhood offsets to use, default is 12.

        mongo_port (``integer``):
            Port number where a MongoDB server instance is listening.

        db_name (``string``):
            Name of the specified MongoDB database to use at the RAG.

        use_mongo (``bool``):
            Flag denoting whether to use a MongoDB RAG or a file-based NetworkX RAG.
    Returns:
        ``bool``:
            Returns ``true`` if all Daisy tasks complete successfully.
    """

    logging.info("Reading affs and fragments")

    affs: Array = open_ds(affs_file, affs_dataset, mode="r")

    chunk_shape: tuple = affs.chunk_shape[affs.n_channel_dims :]

    # task params
    voxel_size: tuple = affs.voxel_size
    read_roi_voxels: Roi = Roi((0, 0, 0), chunk_shape).grow(context, context)
    write_roi_voxels: Roi = Roi((0, 0, 0), chunk_shape)
    total_roi: Roi = affs.roi.grow(context * voxel_size, context * voxel_size)

    read_roi: Roi = read_roi_voxels * voxel_size
    write_roi: Roi = write_roi_voxels * voxel_size

    fragments: Array = open_ds(fragments_file, fragments_dataset, mode="r+")

    # open RAG DB
    if use_mongo:
        db_host: str = f"mongodb://localhost:{mongo_port}"

        logging.info("Opening MongoDBGraphProvider...")
        rag_provider = graphs.MongoDbGraphProvider(
            db_name=db_name,
            host=db_host,
            mode="r+",
            directed=False,
            nodes_collection=sample_name + "_nodes",
            edges_collection=sample_name + "_edges_" + merge_function,
            position_attribute=["center_z", "center_y", "center_x"],
        )
    else:
        logging.info("Opening FileGraphProvider...")
        rag_provider = graphs.FileGraphProvider(
            directory="./RAG",
            mode="r+",
            directed=False,
            nodes_collection=sample_name + "_nodes",
            edges_collection=sample_name + "_edges_" + merge_function,
            position_attribute=["center_z", "center_y", "center_x"],
        )
    logging.info("Graph Provider opened")

    # open block done DB
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    completed_collection_name: str = f"{sample_name}_supervox_blocks_completed"
    completed_collection = db[completed_collection_name]

    # worker func
    def generate_super_voxel_edges_worker(
        block: daisy.Block,
        affs=affs,
        fragments=fragments,
        rag_provider=rag_provider,
        completed_collection=completed_collection,
    ) -> tuple:
        logger.info("getting block")
        start: float = time.time()
        logger.info(
            "Agglomerating in block %s with context of %s",
            block.write_roi,
            block.read_roi,
        )
        logger.info("block read roi begin: %s", block.read_roi.get_begin())
        logger.info("block read roi shape: %s", block.read_roi.get_shape())
        logger.info("block write roi begin: %s", block.write_roi.get_begin())
        logger.info("block write roi shape: %s", block.write_roi.get_shape())

        # get the sub-{affs, fragments, graph} to work on
        affs = affs.intersect(block.read_roi)
        fragments: np.ndarray = fragments.to_ndarray(affs.roi, fill_value=0)
        fragment_ids: np.ndarray = np.array(
            [x for x in np.unique(fragments) if x != 0]
        )
        num_frags: int = len(fragment_ids)
        frag_mapping: dict = {
            old: seq for seq, old in zip(range(1, num_frags + 1), fragment_ids)
        }
        rev_frag_mapping: dict = {
            seq: old for seq, old in zip(range(1, num_frags + 1), fragment_ids)
        }
        for old, seq in frag_mapping.items():
            fragments[fragments == old] = seq
        rag = rag_provider[affs.roi]
        if len(fragment_ids) == 0:
            return

        logger.debug("affs shape: %s", affs.shape)
        logger.debug("fragments shape: %s", fragments.shape)
        # logger.debug("fragments num: %d", n)

        # convert affs to float32 ndarray with values between 0 and 1
        offsets: list[list[int]] = neighborhood[:neighborhood_length]

        affs = affs.to_ndarray()
        if affs.dtype == np.uint8:
            affs: np.float32 = affs.astype(np.float32) / 255.0

        # COMPUTE EDGE SCORES
        # mutex watershed has shown good results when using short range edges
        # for merging objects and long range edges for splitting. So we compute
        # these scores separately

        # separate affinities and offsets by range
        adjacents: list = [offset for offset in offsets if max(offset) <= 1]
        lr_offsets: list = offsets[len(adjacents) :]
        if len(lr_offsets) == 0:
            lr_offsets: list = [[0, 0, 0]]
        affs, lr_affs = affs[: len(adjacents)], affs[len(adjacents) :]
        if lr_bias_ratio != 0:
            for i, offset in enumerate(lr_offsets):
                lr_affs[i] += np.linalg.norm(offset) * lr_bias_ratio

        # COMPUTE EDGE SCORES FOR ADJACENT FRAGMENTS
        max_offset: list = [max(axis) for axis in zip(*adjacents)]
        base_fragments = np.expand_dims(
            fragments[tuple(slice(0, -m) for m in max_offset)], 0
        )
        base_affs = affs[
            (slice(None, None),) + tuple(slice(0, -m) for m in max_offset)
        ]
        offset_frags = []
        for offset in adjacents:
            offset_frags.append(
                fragments[
                    tuple(
                        slice(o, (-m + o) if m != o else None)
                        for o, m in zip(offset, max_offset)
                    )
                ]
            )

        offset_frags: np.ndarray = np.stack(offset_frags, axis=0)
        mask = offset_frags != base_fragments

        # cantor pairing function
        mismatched_labels = (
            (offset_frags + base_fragments)
            * (offset_frags + base_fragments + 1)
            // 2
            + base_fragments
        ) * mask
        mismatched_ids: np.ndarray = np.array(
            [x for x in np.unique(mismatched_labels) if x != 0]
        )
        adjacent_score: float = measurements.median(
            base_affs,
            mismatched_labels,
            mismatched_ids,
        )
        adjacent_map: dict = {
            seq_id: float(med_score)
            for seq_id, med_score in zip(mismatched_ids, adjacent_score)
        }

        # COMPUTE LONG RANGE EDGE SCORES
        max_lr_offset: list = [max(axis) for axis in zip(*lr_offsets)]
        base_lr_fragments = fragments[tuple(slice(0, -m) for m in max_lr_offset)]
        base_lr_affs = lr_affs[
            (slice(None, None),) + tuple(slice(0, -m) for m in max_lr_offset)
        ]
        lr_offset_frags = []
        logger.info(len(lr_offsets))
        for offset in lr_offsets:
            lr_offset_frags.append(
                fragments[
                    tuple(
                        slice(o, (-m + o) if m != o else None)
                        for o, m in zip(offset, max_lr_offset)
                    )
                ]
            )
        lr_offset_frags = np.stack(lr_offset_frags, axis=0)
        lr_mask = lr_offset_frags != base_lr_fragments
        # cantor pairing function
        lr_mismatched_labels = (
            (lr_offset_frags + base_lr_fragments)
            * (lr_offset_frags + base_lr_fragments + 1)
            // 2
            + base_lr_fragments
        ) * lr_mask
        lr_mismatched_ids: np.ndarray = np.array(
            [x for x in np.unique(lr_mismatched_labels) if x != 0]
        )
        lr_adjacent_score = measurements.median(
            base_lr_affs,
            lr_mismatched_labels,
            lr_mismatched_ids,
        )
        lr_adjacent_map: dict = {
            seq_id: float(med_score)
            for seq_id, med_score in zip(lr_mismatched_ids, lr_adjacent_score)
        }

        for seq_id_u, seq_id_v in itertools.combinations(
            range(1, num_frags + 1), 2
        ):
            cantor_id_u: int = (
                (seq_id_u + seq_id_v) * (seq_id_u + seq_id_v + 1)
            ) // 2 + seq_id_u
            cantor_id_v: int = (
                (seq_id_u + seq_id_v) * (seq_id_u + seq_id_v + 1)
            ) // 2 + seq_id_v
            if (
                cantor_id_u in adjacent_map
                or cantor_id_v in adjacent_map
                or cantor_id_u in lr_adjacent_map
                or cantor_id_v in lr_adjacent_map
            ):
                adj_weight_u = adjacent_map.get(cantor_id_u, None)
                adj_weight_v = adjacent_map.get(cantor_id_v, None)
                if adj_weight_u is not None and adj_weight_v is not None:
                    adj_weight = (adj_weight_v + adj_weight_u) / 2
                    # adj_weight += 0.5
                elif adj_weight_u is not None:
                    adj_weight: float = adj_weight_u
                    # adj_weight += 0.5
                elif adj_weight_v is not None:
                    adj_weight: float = adj_weight_v
                    # adj_weight += 0.5
                else:
                    adj_weight = None
                lr_weight_u = lr_adjacent_map.get(cantor_id_u, None)
                lr_weight_v = lr_adjacent_map.get(cantor_id_v, None)
                if lr_weight_u is None and lr_weight_v is None:
                    lr_weight = None
                elif lr_weight_u is None:
                    lr_weight = lr_weight_v
                elif lr_weight_v is None:
                    lr_weight = lr_weight_u
                else:
                    lr_weight: float = (lr_weight_u + lr_weight_v) / 2
                rag.add_edge(
                    rev_frag_mapping[seq_id_u],
                    rev_frag_mapping[seq_id_v],
                    adj_weight=adj_weight,
                    lr_weight=lr_weight,
                )

        # write back results (only within write_roi)
        logger.info(f"writing {len(rag.edges)} edges to DB...")
        logger.info(
            f"num frags: {len(fragment_ids)}, num_adj: {len(adjacent_map)}, "
            f"num_lr_adj: {len(lr_adjacent_map)}"
        )
        rag.write_edges(block.write_roi)

        document: dict = {
            "block_id": block.block_id,
            "read_roi": (block.read_roi.get_begin(), block.read_roi.get_shape()),
            "write_roi": (block.write_roi.get_begin(), block.write_roi.get_shape()),
            "start": start,
            "duration": time.time() - start,
        }

        # add block to completed array
        completed_collection.insert_one(document=document)

        logger.info(f"block information: {document}")
        logger.info(f"releasing block: {block}")
        return True

    # create Daisy distributed task
    task: daisy.Task = daisy.Task(
        "GenSupervoxelEdgesTask",
        total_roi=total_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=generate_super_voxel_edges_worker,
        num_workers=nworkers,
    )

    # run task blockwise
    ret: bool = daisy.run_blockwise([task])
    return ret
