#!/bin/bash

module load gcc/7.3.1
module load cmake/3.14.5
module load cuda/11.7.0

# regent=/usr/workspace/wsb/wei8/dlb/legion/language/regent.py
regent=/usr/workspace/wsb/wei8/release_exp/legion/language/regent.py

build_option="-fdebug 0"
SAVEOBJ=1 STANDALONE=1 OBJNAME=./dlb $regent dlb.rg $build_option
