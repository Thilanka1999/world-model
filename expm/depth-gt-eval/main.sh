#! /bin/bash

python mt_pipe/singlestage.py -c expm/depth-gt-eval/configs/depth-gt-eval.yaml --mb 1 --me 1
#  --ckpt-path models/mock.ckpt -r 1