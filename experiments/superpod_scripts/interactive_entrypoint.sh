#!/usr/bin/env zsh

srun\
    -N1\
    -G1\
    -p short\
    --no-container-entrypoint\
    --container-image /work/group/humingamelab/sqsh_images/nvidia-pyg.sqsh\
    --container-mounts="${HOME}"/Projects/het-jepa:/het-jepa,/work/users/hunjael/data/het-jepa_data/:/data,"${SCRATCH}":/models\
    --container-workdir /het-jepa\
    --pty bash -i