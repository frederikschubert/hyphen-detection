#!/bin/zsh

#SBATCH --partition=gpu_cluster_enife
#SBATCH --mail-type=ALL
#SBATCH --mail-user=schubert@tnt.uni-hannover.de
#SBATCH --time=7-0
#SBATCH --output=/home/schubert/projects/hyphen/nobackup/slurm_logs/%x-%j.slurm.log
#SBATCH --export=ALL
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=3G
#SBATCH --gpus=2

mkdir -p /localstorage/schubert/wandb/
cd /home/schubert/projects/hyphen
. /home/schubert/miniconda3/tmp/bin/activate hyphen

srun $@