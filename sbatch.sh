#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 40
#SBATCH -p cpu

export LD_LIBRARY_PATH="$PWD"

GASNET_BACKTRACE=1 mpirun --bind-to none dlb -ll:gpu 0 -ll:cpu 4 -ll:csize 150000 -level mapper=debug -logfile mapper_ori%.log -lg:prof 1 -lg:prof_logfile prof_%.gz
