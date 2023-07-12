# rusty_mws

[![](https://img.shields.io/pypi/pyversions/mwatershed.svg)](https://pypi.python.org/pypi/mwatershed)
[![](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


## An optimized implementation of Rust-based Mutex Watershed image segmentation, built on MongoDB.



* Free software: Apache 2.0 License

### Installation

Install Rust and Cargo via RustUp:

```bash
curl https://sh.rustup.rs -sSf | sh
```


Install MongoDB:

```bash
curl -fsSL https://pgp.mongodb.com/server-6.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-6.0.gpg --dearmor
```


And initialize a MongoDB server in a screen on your machine:

```bash
screen
```

```bash
mongod
```


Install `rusty_mws`:

```bash
pip install git+https://github.com/brianreicher/rusty_mws.git
```

### Features

* A mutex watershed implementation for affinities
* Should handle up to 6 dimensional arrays of fragments (7 dimensions if you include offsets in affinities) but only tested on 2 and 3 dimensions

### Usage

For seedless Mutex Watershed segmentation, using MongoDB:
```python
_: bool = rusty_mws.run_pred_segmentation_pipeline(
            affs_file="../data/raw_predictions.zarr",
            affs_dataset="pred_affs_latest",
            fragments_file="../data/raw_predictions.zarr",
            fragments_dataset="frag_seg",
            context=Coordinate(np.max(a=np.abs(rusty_mws.neighborhood), axis=0)),
        )
```
where:
* `affs_file` is a path (relative or absolute) to the zarr file containing predicted affinities to generate fragments for.
* `affs_dataset` is the name of the affinities dataset in the affs_file to read from.
* `fragments_file` is a path (relative or absolute) to the zarr file to write fragments to.
* `fragments_dataset` is the name of the fragments dataset to read/write to in the fragments_file.
* `context` is a 3-dimensional coordinate object denoting how much contextual space to grow for the total volume ROI during segmentation.

For seeded skeleton-corrected segmentation with Mutex Watershed fragments:
```python
_: bool = rusty_mws.run_corrected_segmentation_pipleine(
            affs_file="../data/raw_predictions.zarr",
            affs_dataset="pred_affs_latest",
            fragments_file="../data/raw_predictions.zarr",
            fragments_dataset="frag_seg",
            seeds_file="../data/raw_predictions.zarr",
            seeds_dataset="training_gt_rasters",
            context=Coordinate(np.max(a=np.abs(rusty_mws.neighborhood), axis=0)),
        )
```
where:
* `affs_file` is a path (relative or absolute) to the zarr file containing predicted affinities to generate fragments for.
* `affs_dataset` is the name of the affinities dataset in the affs_file to read from.
* `fragments_file` is a path (relative or absolute) to the zarr file to write fragments to.
* `fragments_dataset` is the name of the fragments dataset to read/write to in the fragments_file.
* `seeds_file` is a path (relative or absolute) to the zarr file containing seeds.
* `seeds_dataset` is the name of the seeds dataset in the seeds file to read from.
* `context` is a 3-dimensional coordinate object denoting how much contextual space to grow for the total volume ROI during segmentation.

### Credits

This package builds upon [`mwatershed`](https://github.com/pattonw/mwatershed), developed by William Patton.