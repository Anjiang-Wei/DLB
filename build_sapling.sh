#!/bin/bash

regent=/scratch2/anjiang/DSLMapperExp/legion/language/regent.py

build_option="-findex-launch 1 -fdebug 1"
SAVEOBJ=1 STANDALONE=1 OBJNAME=./dlb $regent dlb.rg $build_option

if [ $? -eq 0 ]; then
    sbatch sbatch.sh
fi