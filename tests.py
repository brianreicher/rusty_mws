import rusty_mws


pp: rusty_mws.PostProcessor = rusty_mws.PostProcessor(
            affs_file="./data/raw_predictions.zarr",
            affs_dataset="pred_affs_latest",
        )
pp.run_pred_segmentation_pipeline()