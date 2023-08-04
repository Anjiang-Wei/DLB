#!/bin/bash

regent=/usr/workspace/wsb/wei8/dlb/legion/language/regent.py

build_option="-findex-launch 1 -fdebug 1"
SAVEOBJ=1 STANDALONE=1 OBJNAME=./dlb $regent dlb.rg $build_option