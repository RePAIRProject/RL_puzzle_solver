#!/bin/bash

#SBATCH --job-name=im00to04_50lines_puzzle_64anchors		   		# Job name
#SBATCH --ntasks=1                 				                    # Run on a single CPU
#SBATCH --cpus-per-task=1
#SBATCH --output=RL_logs/64anchors_50lines_img00to04.log			    # Standard output and error log

# PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:/home/luca.palmieri/code/RL_puzzle_solver

dataset='images_with_50_lines'
pieces=8

# CONDA ENVIRONMENT
#conda activate repair
for puzzle in 'image_0' 'image_1' 'image_2' 'image_3' 'image_4' 
do
    echo "IMAGE $puzzle"
    #echo "compatibility.."
    #python compatibility/line_matching_NEW_segments.py --dataset $dataset --puzzle $puzzle --pieces $pieces #--penalty $3

    echo "solver.."
    for anchor in {0..64}
    do 
        echo "solver with anchor $anchor"
        python solver/solverRotPuzzArgs.py --dataset $dataset --puzzle $puzzle --pieces $pieces --anchor $anchor --verbosity 0

        echo "evaluating with anchor $anchor"
        python metrics/evaluate.py --dataset $dataset --puzzle $puzzle --num_pieces $pieces --anchor $anchor

    done
done
