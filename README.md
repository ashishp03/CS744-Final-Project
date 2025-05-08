# CS744-Final-Project

## Introduction
We present our code for training the mt5_small on the M2Lingual dataset. There were different versions of the training done depending on whether the setup was distributed or run on a single node. The M2Lingual dataset is made available on HuggingFace [here](https://huggingface.co/datasets/ServiceNow-AI/M2Lingual).

## Requirements and instructions for running
### Single Node setup
The files `mt5_single_seed_p100.py` and `mt5_single_full_a100` can be run on one single node with the simple command `python3 [file_name.py]`
### Distributed Setup
Need to have a VM with a cluster of at least 2 GPUs. The files `mt5_distributed_seed_p100.py` and `mt5_distributed_full_p100.py` can be run in this way: `python3 [file_name.py] --rank [node i] --world_size [number of GPUs available in your cluster]`   
Run this command on each node in your cluster, incrementing i per node. i  will range from 0 to (world_size - 1). The final aggregation and generation happens on node 0.