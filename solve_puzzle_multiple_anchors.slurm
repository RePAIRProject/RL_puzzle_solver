#!/bin/bash

#SBATCH --job-name=64anchors_50lines_puzzle		   		# Job name
#SBATCH --ntasks=64                 				    # Run on a single CPU
#SBATCH --output=RL_logs/64anchors_50lines_img0.log			    # Standard output and error log

# PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:/home/luca.palmieri/code/RL_puzzle_solver

dataset='images_with_50_lines'
puzzle='image_1'
pieces=8


# CONDA ENVIRONMENT
#conda activate repair

echo "compatibility.."
python compatibility/line_matching_NEW_segments.py --dataset $dataset --puzzle $puzzle --pieces $pieces #--penalty $3

echo "solver.."
for anchor in {0..64}
do 
    echo "solver with anchor $anchor"
    python solver/solverRotPuzzArgs.py --dataset $dataset --puzzle $puzzle --pieces $pieces --anchor $anchor

    echo "evaluating with anchor $anchor"
    python metrics/evaluate.py --dataset $dataset --puzzle $puzzle --num_pieces $pieces --anchor $anchor

done
