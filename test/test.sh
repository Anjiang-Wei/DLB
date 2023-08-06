#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 40
#SBATCH -p cpu

# export LD_LIBRARY_PATH="/scratch2/anjiang/DSLMapperExp/legion/bindings/regent:$PWD"

GASNET_BACKTRACE=1 mpirun --bind-to none legion_tasksteal_test -ll:cpu 4 -lg:prof 1  -lg:prof_logfile prof_%.gz
