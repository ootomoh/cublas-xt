#!/bin/sh
#PBS -N cublasxt
#PBS -M o.h.kisaragi@gmail.com
#PBS -m e
cd ${PBS_O_WORKDIR}

./exec
./exec_xt
