#!/bin/bash

#SBATCH --job-name=noisy_solver                                       # Job name
#SBATCH --ntasks=1                                                      # Run on a single CPU
#SBATCH --output=RL_logs/noise/noisy_%j.log                 # Standard output and error log

# PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:/home/luca.palmieri/code/RL_puzzle_solver

dataset=$1
pieces=8
from=$2
to=$3

# CONDA ENVIRONMENT
#conda activate repair
for puzzle_num in {$from..$to}
do
    
    echo "image_$puzzle_num"
    echo "compatibility.."
    python compatibility/line_matching_NEW_segments.py --dataset $dataset --puzzle image_$puzzle_num --pieces $pieces --method deeplsd #--penalty $3
    python compatibility/line_matching_NEW_segments.py --dataset $dataset --puzzle image_$puzzle_num --pieces $pieces --method noise_positional_p30_s1 #--penalty $3
    python compatibility/line_matching_NEW_segments.py --dataset $dataset --puzzle image_$puzzle_num --pieces $pieces --method noise_positional_p30_s2 #--penalty $3
    python compatibility/line_matching_NEW_segments.py --dataset $dataset --puzzle image_$puzzle_num --pieces $pieces --method noise_positional_p30_s3 #--penalty $3
    echo "solver DEEPLSD.."
    python solver/solverRotPuzzArgs.py --dataset $dataset --puzzle image_$puzzle_num --pieces $pieces --method deeplsd
    echo "solver NOISE S1.."
    python solver/solverRotPuzzArgs.py --dataset $dataset --puzzle image_$puzzle_num --pieces $pieces --method noise_positional_p30_s1
    echo "solver NOISE S2.."
    python solver/solverRotPuzzArgs.py --dataset $dataset --puzzle image_$puzzle_num --pieces $pieces --method noise_positional_p30_s2
    echo "solver NOISE S3.."
    python solver/solverRotPuzzArgs.py --dataset $dataset --puzzle image_$puzzle_num --pieces $pieces --method noise_positional_p30_s3

    echo "evaluating.."
    python metrics/evaluate_noisy_lines.py --dataset $dataset --puzzle $puzzle --num_pieces $pieces 
done