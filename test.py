import rusty_mws

pp: rusty_mws.PostProcessor = rusty_mws.PostProcessor(
            affs_file="./data/raw_predictions.zarr",
            affs_dataset="pred_affs_latest",
            seeds_file="./data/raw_predictions.zarr",
            seeds_dataset="training_gt_rasters",
        )
pp.run_corrected_segmentation_pipeline()
