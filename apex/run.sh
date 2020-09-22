#!/bin/bash

python -m torch.distributed.launch --nproc_per_node 3 test1.py
