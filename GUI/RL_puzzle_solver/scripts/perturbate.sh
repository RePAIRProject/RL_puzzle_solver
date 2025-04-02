#!/bin/bash

#SBATCH --job-name=select_anchors		   		                        # Job name
#SBATCH --ntasks=1                 				                        # Run on a single CPU
#SBATCH --cpus-per-task=1
#SBATCH --output=RL_logs/anchors_selection.log			# Standard output and error log

# PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:/home/luca.palmieri/code/RL_puzzle_solver

python datasets/perturbate_dataset.py -m exact -n structural -p 10