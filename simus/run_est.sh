#!/bin/bash
#SBATCH --job-name=Splitting_FBM_055
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
##SBATCH --mem-per-cpu=8000
#SBATCH --no-requeue
#SBATCH --partition=normal
#SBATCH --array=1-35

echo "SLURM: $SLURM_JOB_NODELIST, $SLURM_TASKS_PER_NODE"
nice -19 python3 SplittingFBMx0.py $SLURM_ARRAY_TASK_ID 
