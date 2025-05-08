# CS744-Final-Project

## Introduction
We present our code for training the mt5_small on the M2Lingual dataset. There were different versions of the training done depending on whether the setup was distributed or run on a single node. The M2Lingual dataset is made available on HuggingFace [here](https://huggingface.co/datasets/ServiceNow-AI/M2Lingual).

## Requirements and instructions for running
### Dataset instructions
When running the files be careful to use the seed or full_data configs of the M2Lingual dataset. The file names indicate whether the seed data or the full_data is being used. These can't be used interchangably as the full_data contains additional features which the seed data doesn't have. Therefore, when switching, the `*-full-*.py` files always use the full dataset will the `*-seed.py-*` files always use the seed data.
### Single Node setup
The files `mt5_single_seed_p100.py` and `mt5_single_full_a100` can be run on one single node with the simple command `python3 [file_name.py]`
### Distributed Setup
Need to have a VM with a cluster of at least 2 GPUs. The files `mt5_distributed_seed_p100.py` and `mt5_distributed_full_p100.py` can be run in this way: `python3 [file_name.py] --rank [node i] --world_size [number of GPUs available in your cluster]`   
Run this command on each node in your cluster, incrementing i per node. i  will range from 0 to (world_size - 1). The final aggregation and generation happens on node 0.