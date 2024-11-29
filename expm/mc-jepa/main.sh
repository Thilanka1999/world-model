#! /bin/bash

bash mt_pipe/multistage.sh\
    -c expm/mc-jepa/configs/content_flow.yaml expm/mc-jepa/configs/content.yaml\
    -t expm/mc-jepa/configs/stage_trans.yaml\
    --run-name mc-jepa\
    --use-amp 1\
    --mock-batch-count 1 --mock-epoch-count 1