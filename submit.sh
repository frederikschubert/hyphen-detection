#!/bin/zsh

#SBATCH --partition=gpu_cluster_enife
#SBATCH --mail-type=ALL
#SBATCH --mail-user=schubert@tnt.uni-hannover.de
#SBATCH --time=7-0
#SBATCH --output=/home/schubert/projects/hyphen/nobackup/slurm_logs/%x-%j.slurm.log
#SBATCH --export=ALL
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3G
#SBATCH --gpus=2


cd /home/schubert/projects/hyphen
. /home/schubert/miniconda3/tmp/bin/activate hyphen

srun $@