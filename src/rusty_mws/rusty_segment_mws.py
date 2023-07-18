import logging
import time
import os
from funlib.geometry import Coordinate
from typing import Optional
from algo.generate_mutex_fragments import *
from algo.generate_supervoxel_edges import *
from algo.global_mutex_agglom import *
from algo.extract_seg_from_luts import *
from algo.skeleton_correct import *


logger: logging.Logger = logging.getLogger(name=__name__)

class PostProcessor():
    """Driver for post-processing segmentation.

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
            
            adj_bias (``float``):
                Amount to bias adjacent pixel weights when computing segmentation from the stored graph.

            lr_bias (``float``):
                Amount to bias long-range pixel weights when computing segmentation from the stored graph.

            nworkers_supervox (``integer``):
                Number of distributed workers to run the Daisy parallel supervoxel task with.
            
            merge_function (``str``):
                Name of the segmentation algorithm used to denote in the MongoDB edge collection.
            
            nworkers_lut (``integer``):
                Number of distributed workers to run the Daisy parallel LUT extraction task with.

            n_chunk_write_lut (``integer``):
                Number of chunks to write for each Daisy block in the LUT extraction task.

    """

    def __init__(self,    
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
                adj_bias: float = -0.1,
                lr_bias: float = -1.5,
                nworkers_supervox: int = 25,
                merge_function: str = "mwatershed",
                nworkers_lut: int = 25,
                n_chunk_write_lut: int = 1,) -> None:
        
        # dataset vars
        self.affs_file: str = affs_file
        self.affs_dataset: str = affs_dataset
        self.fragments_file: str = fragments_file
        self.fragments_dataset: str = fragments_dataset
        self.seeds_file: str = seeds_file
        self.seeds_dataset: str = seeds_dataset
        self.seg_file: str = seg_file
        self.seg_dataset: str = seg_dataset
        self.mask_file: str = mask_file
        self.mask_dataset: str = mask_dataset

        # dataset processing vars
        self.context: Coordinate = context
        self.filter_val: float = filter_val
        self.seeded: bool = seeded
        self.merge_function: str = merge_function

        # MongoDB vars
        self.mongo_port: int = mongo_port
        self.db_name: str = db_name
        self.sample_name: str = sample_name

        # Daisy vars
        self.nworkers_frags: int = nworkers_frags
        self.nworkers_correct: int = nworkers_correct
        self.nworkers_supervox: int  = nworkers_supervox
        self.nworkers_lut: int = nworkers_lut
        self.n_chunk_write_frags: int = n_chunk_write_frags
        self.n_chunk_write_correct: int - n_chunk_write_correct
        self.n_chunk_write_lut: int = n_chunk_write_lut

        # fragment weight and neighborhood vars
        self.lr_bias_ratio: float = lr_bias_ratio
        self.adjacent_edge_bias: float = adjacent_edge_bias
        self.neighborhood_length: int = neighborhood_length

        # skeleton correction vars
        self.erode_iterations: int = erode_iterations
        self.erode_footprint: np.ndarray = erode_footprint
        self.alternate_dilate: bool = alternate_dilate
        self.dilate_footprint: np.ndarray = dilate_footprint

        # MWS agglom vars
        self.adj_bias: float = adj_bias
        self.lr_bias: float = lr_bias

    def run_corrected_segmentation_pipeline(self,) -> bool:
        """Full skeleton-corrected MWS segmentation from affinities.

        Returns:
            ``bool``:
                Denotes whether or not the segmentation is completed successfully.
        """

        if self.sample_name is None:
            self.sample_name: str = "htem" + str(
                hash(
                    f"FROM{os.path.join(self.affs_file, self.affs_dataset)}TO{os.path.join(self.fragments_file, self.fragments_dataset)}AT{time.strftime('%Y%m%d-%H%M%S')}".replace(
                        ".", "-"
                    ).replace(
                        "/", "-"
                    )
                )
            )

        success: bool = True
        if self.seeded:
            success = success & blockwise_generate_mutex_fragments(
                sample_name=self.sample_name,
                affs_file=self.affs_file,
                affs_dataset=self.affs_dataset,
                fragments_file=self.fragments_file,
                fragments_dataset=self.fragments_dataset,
                context=self.context,
                filter_val=self.filter_val,
                seeds_file=self.seeds_file,
                seeds_dataset=self.seeds_dataset,
                mask_file=self.mask_file,
                mask_dataset=self.mask_dataset,
                training=True,
                nworkers=self.nworkers_frags,
                n_chunk_write=self.n_chunk_write_frags,
                lr_bias_ratio=self.lr_bias_ratio,
                adjacent_edge_bias=self.adjacent_edge_bias,
                neighborhood_length=self.neighborhood_length,
                mongo_port=self.mongo_port,
                db_name=self.db_name,
            )
        else:
            success = success & blockwise_generate_mutex_fragments(
                sample_name=self.sample_name,
                affs_file=self.affs_file,
                affs_dataset=self.affs_dataset,
                fragments_file=self.fragments_file,
                fragments_dataset=self.fragments_dataset,
                context=self.context,
                filter_val=self.filter_val,
                seeds_file=None,
                seeds_dataset=None,
                training=True,
                mask_file=self.mask_file,
                mask_dataset=self.mask_dataset,
                training=True,
                nworkers=self.nworkers_frags,
                n_chunk_write=self.n_chunk_write_frags,
                lr_bias_ratio=self.lr_bias_ratio,
                adjacent_edge_bias=self.adjacent_edge_bias,
                neighborhood_length=self.neighborhood_length,
                mongo_port=self.mongo_port,
                db_name=self.db_name,
            )

        success = success & skel_correct_segmentation(
            seeds_file=self.seeds_file,
            seeds_dataset=self.seeds_dataset,
            frag_file=self.fragments_file,
            frag_name=self.fragments_dataset,
            seg_file=self.seg_file,
            seg_dataset=self.seg_dataset,
            nworkers=self.nworkers_correct,
            erode_iterations=self.erode_iterations,
            erode_footprint=self.erode_footprint,
            alternate_dilate=self.alternate_dilate,
            dilate_footprint=self.dilate_footprint,
            n_chunk_write=self.n_chunk_write_correct,
        )

        return success


    def run_pred_segmentation_pipeline(self,) -> bool:
        """Full Mutex Watershed segmentation and agglomeration, using a MongoDB RAG.

        Returns:
            ``bool``:
                Denotes whether or not the segmentation is completed successfully.
        """

        if self.sample_name is None:
            self.sample_name: str = "htem" + str(
                hash(
                    f"FROM{os.path.join(self.affs_file, self.affs_dataset)}TO{os.path.join(self.fragments_file, self.fragments_dataset)}AT{time.strftime('%Y%m%d-%H%M%S')}".replace(
                        ".", "-"
                    ).replace(
                        "/", "-"
                    )
                )
            )

        success: bool = True

        success = success & blockwise_generate_mutex_fragments(
                sample_name=self.sample_name,
                affs_file=self.affs_file,
                affs_dataset=self.affs_dataset,
                fragments_file=self.fragments_file,
                fragments_dataset=self.fragments_dataset,
                context=self.context,
                filter_val=self.filter_val,
                seeds_file=None,
                seeds_dataset=None,
                training=True,
                mask_file=self.mask_file,
                mask_dataset=self.mask_dataset,
                training=False,
                nworkers=self.nworkers_frags,
                n_chunk_write=self.n_chunk_write_frags,
                lr_bias_ratio=self.lr_bias_ratio,
                adjacent_edge_bias=self.adjacent_edge_bias,
                neighborhood_length=self.neighborhood_length,
                mongo_port=self.mongo_port,
                db_name=self.db_name,
        )

        success = success & blockwise_generate_supervoxel_edges(
            sample_name=self.sample_name,
            affs_file=self.affs_file,
            affs_dataset=self.affs_dataset,
            fragments_file=self.fragments_file,
            fragments_dataset=self.fragments_dataset,
            context=self.context,
            nworkers=self.nworkers_supervox,
            merge_function=self.merge_function,
            lr_bias_ratio=self.lr_bias_ratio,
            neighborhood_length=self.neighborhood_length,
            mongo_port=self.mongo_port,
            db_name=self.db_name,
        )

        success = success & global_mutex_agglomeration(
            sample_name=self.sample_name,
            fragments_file=self.fragments_file,
            fragments_dataset=self.fragments_dataset,
            merge_function=self.merge_function,
            adj_bias=self.adj_bias,
            lr_bias=self.lr_bias,
            mongo_port=self.mongo_port,
            db_name=self.db_name,
        )

        success = success & extract_segmentation(
            fragments_file=self.fragments_file,
            fragments_dataset=self.fragments_dataset,
            nworkers=self.nworkers_lut,
            merge_function=self.merge_function,
            n_chunk_write=self.n_chunk_write_lut,
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
