#!/bin/bash
#SBATCH --job-name=dli_assessment_step5
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1       
#SBATCH --cpus-per-task=32 ### Number of threads per task (OMP threads)
#SBATCH -o /dli/megatron/logs/%j.out
#SBATCH -e /dli/megatron/logs/%j.err

# Number of nodes
NUM_NODES=2
# Number of GPUs per node
NUM_GPUS=2

deepspeed --num_nodes=${NUM_NODES} --hostfile /dli/minGPT/minGPT/hostfile --num_gpus=${NUM_GPUS} /dli/minGPT/minGPT/runStep5.py \
    --deepspeed \
    --deepspeed_config /dli/minGPT/minGPT/ds_config_step5.json
