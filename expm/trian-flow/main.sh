#! /bin/bash

# python mt_pipe/singlestage.py -c expm/trian-flow/configs/ssl-flow.yaml -d 1
torchrun --nnodes 1 --nproc_per_node 2 mt_pipe/singlestage.py -c expm/trian-flow/configs/ssl-flow.yaml -d 0 1