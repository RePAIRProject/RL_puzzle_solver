#!/bin/bash

#SBATCH --job-name=64as_im21_50lines_puzzle		   		# Job name
#SBATCH --ntasks=1                 				        # Run on a single CPU
#SBATCH --cpus-per-task=1
#SBATCH --output=RL_logs/64anchors_50lines_img1.log			    # Standard output and error log

# PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:/home/luca.palmieri/code/RL_puzzle_solver

dataset='images_with_50_lines'
puzzle=$1
pieces=8


# CONDA ENVIRONMENT
#conda activate repair
echo "IMAGE $puzzle"

echo "compatibility.."
python compatibility/line_matching_NEW_segments.py --dataset $dataset --puzzle $puzzle --pieces $pieces #--penalty $3

echo "solver.."
for anchor in {0..64}
do 
    echo "solver with anchor $anchor"
    python solver/solverRotPuzzArgs.py --dataset $dataset --puzzle $puzzle --pieces $pieces --anchor $anchor --verbosity 0

    echo "evaluating with anchor $anchor"
    python metrics/evaluate.py --dataset $dataset --puzzle $puzzle --num_pieces $pieces --anchor $anchor

done
