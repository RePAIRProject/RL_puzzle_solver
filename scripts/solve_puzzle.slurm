#!/bin/bash

#SBATCH --job-name=solver_noisy_lines                                       # Job name
#SBATCH --ntasks=1                                                      # Run on a single CPU
#SBATCH --output=RL_logs/exp_single_puzzle_noisy_%j.log                 # Standard output and error log

# PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:/home/luca.palmieri/code/RL_puzzle_solver

dataset=$1
puzzle=$2
pieces=$3
method=$4

# CONDA ENVIRONMENT
#conda activate repair

echo "compatibility.."
python compatibility/line_matching_NEW_segments.py --dataset $dataset --puzzle $puzzle --pieces $pieces --method $method #--penalty $3

echo "solver.."
python solver/solverRotPuzzArgs.py --dataset $dataset --puzzle $puzzle --pieces $pieces

echo "evaluating.."
python metrics/evaluate.py --dataset $dataset --puzzle $puzzle --num_pieces $pieces 
