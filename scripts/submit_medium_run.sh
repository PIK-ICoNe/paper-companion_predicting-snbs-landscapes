#!/bin/bash

#SBATCH --qos=gpumedium
#SBATCH --job-name=runbo17_1
#SBATCH --output=logs/%x-%j-%N.out
#SBATCH --error=logs/%x-%j-%N.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=80000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --time=4-0
export TEMP=$TMPDIR
export TEMPDIR=$TMPDIR
export TMPDIR=/p/tmp/nauck/optuna/$SLURM_JOBID;
mkdir $TMPDIR

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"

module use /p/system/modulefiles/compiler /p/system/modulefiles/gpu /p/system/modulefiles/library /p/system/modulefiles/tools
module use cuda
module load anaconda
source activate py312ptg

# NOTEBOOKPORT=`shuf -i 8000-8500 -n 1`
# TUNNELPORT=`shuf -i 8501-9000 -n 1`

# ssh -R$TUNNELPORT:localhost:$NOTEBOOKPORT $SLURM_SUBMIT_HOST -N -f

# echo "ssh -L8887:localhost:$TUNNELPORT $SLURM_SUBMIT_HOST -N"

python run_multiple_seeds.py --seeds 1 2 3 4 5 #$SLURM_JOBID

rm -rf $TMPDIR 
