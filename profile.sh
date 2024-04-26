#!/bin/bash
#SBATCH --export=/usr/local/cuda/bin
#SBATCH --gres=gpu:1

module load cuda/11.5

make main

nvprof --analysis-metrics -o main.nvvp ./build/main


module unload cuda/11.5
