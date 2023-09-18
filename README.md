# rusty_mws

[![](https://img.shields.io/pypi/pyversions/mwatershed.svg)](https://pypi.python.org/pypi/mwatershed)
[![](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


## A parallelized implementation of Rust-based Mutex Watershed image segmentation, built using MongoDB and Daisy for distributed processing.



* Free software: Apache 2.0 License

### Installation

A full installation script can be found in [`install.sh`](https://github.com/brianreicher/rusty_mws/blob/main/install.sh), or line-by-line installs can be ran as follows:

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
mongod
```

Install ``graph_tool``

```bash
conda install -c conda-forge -c ostrokach-forge -c pkgw-forge graph-tool
```


Install `rusty_mws`:

```bash
pip install git+https://github.com/brianreicher/rusty_mws.git
```

### Features

* A mutex watershed implementation to go from affininities to a full segmentation
* Able to run pipelines for skeleton-corrected segmentation (without graph agglomeration) or graph-based Mutex Watershed segmentation
* Should handle up to 6 dimensional arrays of fragments (7 dimensions if you include offsets in affinities) but only tested on 2 and 3 dimensions

### Usage

For seedless Mutex Watershed segmentation, using MongoDB:
```python
import rusty_mws

pp: rusty_mws.PostProcessor = rusty_mws.PostProcessor(
            affs_file="../data/raw_predictions.zarr",
            affs_dataset="pred_affs_latest",
        )
pp.segment_mws()
```
where:
* `affs_file` is a path (relative or absolute) to the zarr file containing predicted affinities to generate fragments for.
* `affs_dataset` is the name of the affinities dataset in the affs_file to read from.

For seeded skeleton-corrected segmentation with Mutex Watershed fragments:
```python
import rusty_mws

pp: rusty_mws.PostProcessor = rusty_mws.PostProcessor(
            affs_file="../data/raw_predictions.zarr",
            affs_dataset="pred_affs_latest",
            seeds_file="../data/raw_predictions.zarr",
            seeds_dataset="training_gt_rasters",
        )
pp.segment_seed_correction()
```
where:
* `affs_file` is a path (relative or absolute) to the zarr file containing predicted affinities to generate fragments for.
* `affs_dataset` is the name of the affinities dataset in the affs_file to read from.
* `seeds_file` is a path (relative or absolute) to the zarr file containing seeds.
* `seeds_dataset` is the name of the seeds dataset in the seeds file to read from.

### Credits

This package builds upon [`mwatershed`](https://github.com/pattonw/mwatershed), developed by William Patton.
