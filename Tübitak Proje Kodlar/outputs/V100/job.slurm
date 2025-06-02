#!/bin/bash
#SBATCH --partition=akya-cuda
#SBATCH --job-name=tubitak-proje
#SBATCH --reservation=yzup-ders
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:4
#SBATCH --output=./out/%x.%j.out #Note : %x == job-name
#SBATCH --error=./out/slurm-%j.err
#SBATCH --account=egitim

mkdir -p ./out

nvidia-smi --query-gpu=timestamp,name,pci.bus_id,power.draw,temperature.gpu,utilization.gpu,utilization.memory --format=csv -l >> ./out/nvidia.out  &

module load apps/truba-ai/gpu-2024.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

batch_sizes=(32 128 512)
optimizers=("adam" "sgd" "sgd-momentum" "rmsprop")
strategies=("ddp" "fsdp" "fsdp1" "fsdp2")

for batch_size in "${batch_sizes[@]}"; do
  for optimizer in "${optimizers[@]}"; do
    for strategy in "${strategies[@]}"; do
      echo "Running with batch_size=$batch_size, optimizer=$optimizer, strategy=$strategy"
      srun python3 deepl.py \
        --gpus=4 \
        --nodes=1 \
        --epochs=5 \
        --batch_size=$batch_size \
        --optimizer=$optimizer \
        --strategy=$strategy
    done
  done
done
exit
