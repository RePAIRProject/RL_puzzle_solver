#!/bin/bash

#SBATCH --job-name=im00_01_50lines_puzzle_64anchors		   		# Job name
#SBATCH --ntasks=1                 				                    # Run on a single CPU
#SBATCH --cpus-per-task=1
#SBATCH --output=RL_logs/64anchors_50lines_img%j.log			    # Standard output and error log

# PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:/home/luca.palmieri/code/RL_puzzle_solver

dataset='images_with_50_lines'
from=$1
to=$2
pieces=8

# CONDA ENVIRONMENT
#conda activate repair
for ((puzzle_num=$1; puzzle_num<$2; puzzle_num++)) 
do
    echo "IMAGE $puzzle"
    #echo "compatibility.."
    #python compatibility/line_matching_NEW_segments.py --dataset $dataset --puzzle $puzzle --pieces $pieces #--penalty $3

    echo "solver.."
    for anchor in {0..64}
    do 
        echo "solver with anchor $anchor"
        python solver/solverRotPuzzArgs.py --dataset $dataset --puzzle image_$puzzle_num --pieces $pieces --anchor $anchor --verbosity 0

        echo "evaluating with anchor $anchor"
        python metrics/evaluate.py --dataset $dataset --puzzle image_$puzzle_num --num_pieces $pieces --anchor $anchor

    done
done
