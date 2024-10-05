#!/usr/bin/env zsh
#SBATCH -J test
#SBATCH -o output/tuning/cora_tuning%j.out
#SBATCH -c 16 --mem=82G   
#SBATCH --nodes=1
#SBATCH -G 1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=NONE     


current_percentage=0.1

save_dir="/lustre/smuexa01/client/users/hunjael/results/het-jepa/${SLURM_JOB_ID}_${dataset}"
mkdir -p ${save_dir}

# set read/write permissions
chmod -R 777 ${save_dir}

cleanup() {
    echo "Moving results to ${WORK}/results/"
    mv ${save_dir} ${WORK}/results/het-jepa
    echo "Removing tmp ${save_dir} from SCRATCH"
    rm -rf ${save_dir}
}

# trap the signal to the main BATCH script here.
sig_handler()
{
    echo "BATCH job received signal"
    echo "waiting for srun process $srun_pid to finish"
    wait
}
trap 'sig_handler' SIGTERM SIGINT SIGCONT 



srun\
    --no-container-entrypoint\
    --container-image /work/group/humingamelab/sqsh_images/nvidia-pyg.sqsh\
    --container-mounts="${HOME}"/Projects/het-jepa:/het-jepa,/work/users/hunjael/data/het-jepa_data/:/data,${save_dir}:/models\
    --container-workdir /het-jepa\
    --signal=TERM@32\
    bash -c "/users/hunjael/Projects/het-jepa/experiments/superpod_scripts/tuner_entrypoint.sh" &

srun_pid=$!
wait $srun_pid


# move results from SCRATCH to $WORK using trap
cleanup
