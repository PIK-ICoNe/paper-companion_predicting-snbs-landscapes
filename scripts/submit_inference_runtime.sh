#!/bin/bash

#SBATCH --qos=gpushort
#SBATCH --job-name=inference_runtime
#SBATCH --output=logs/%x-%j-%N.out
#SBATCH --error=logs/%x-%j-%N.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-02:00

export TEMP=$TMPDIR
export TEMPDIR=$TMPDIR
export TMPDIR=/p/tmp/nauck/inference_runtime/$SLURM_JOBID
mkdir -p "$TMPDIR"

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"

module use /p/system/modulefiles/compiler /p/system/modulefiles/gpu /p/system/modulefiles/library /p/system/modulefiles/tools
module use cuda

uv run --offline inference_runtime.py

rm -rf "$TMPDIR"
