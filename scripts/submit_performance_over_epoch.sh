#!/bin/bash

#SBATCH --qos=gpushort
#SBATCH --output=logs/%x-%j-%N.out
#SBATCH --error=logs/%x-%j-%N.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=80000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --time=24:00:00
export TEMP=$TMPDIR
export TEMPDIR=$TMPDIR
export TMPDIR=/p/tmp/nauck/optuna/$SLURM_JOBID;
#mkdir $TMPDIR
mkdir -p $TMPDIR

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"

module use /p/system/modulefiles/compiler /p/system/modulefiles/gpu /p/system/modulefiles/library /p/system/modulefiles/tools
module use cuda

# NOTEBOOKPORT=`shuf -i 8000-8500 -n 1`
# TUNNELPORT=`shuf -i 8501-9000 -n 1`

# ssh -R$TUNNELPORT:localhost:$NOTEBOOKPORT $SLURM_SUBMIT_HOST -N -f

# echo "ssh -L8887:localhost:$TUNNELPORT $SLURM_SUBMIT_HOST -N"

uv run --offline analyze_performance_over_epoch.py "$@"
## Check if the second argument (study_name) is provided
#if [ -z "$2" ]; then
#    # If study_name is not provided, call the script with only training_dir
#    python eval_multiple_seeds.py --training_dir "$1"
#else
#    # If study_name is provided, include it in the command
##    python eval_multiple_seeds.py --training_dir "$1" --study_name "$2"
#    python eval_multiple_seeds.py --training_dir "$1" --extra_grids "$2"
#
#fi


# python eval_multiple_seeds.py --training_dir "../../ml_training/run_bo_opt13_4"

#rm -rf $TMPDIR 
