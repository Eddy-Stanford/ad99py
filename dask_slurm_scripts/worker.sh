#!/usr/bin/env bash
#SBATCH -J dask-worker-rob
#SBATCH -e /scratch/users/robcking/dask_worker_logs/dask-worker-%J-rob.err
#SBATCH -o /scratch/users/robcking/dask_worker_logs/dask-worker-%J-rob.out
#SBATCH -p serc
#SBATCH -n 1
#SBATCH -C "CLASS:SH3_CBASE|CLASS:SH3_CPERF"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 6:00:00
/home/groups/aditis2/robcking/condaenvs/ad99py/bin/python -m distributed.cli.dask_worker  tcp://10.19.14.17:39188 --name rob-worker-${SLURM_JOB_ID} --nthreads 1 --memory-limit 12GiB --nworkers 8 --nanny --death-timeout 60 
