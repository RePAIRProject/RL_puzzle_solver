#!/bin/bash

#SBATCH --job-name=img16-20_50lines_puzzle_64anchors		   		# Job name
#SBATCH --ntasks=1                 				                    # Run on a single CPU
#SBATCH --cpus-per-task=1
#SBATCH --output=RL_logs/64anchors_50lines_img16to20.log			    # Standard output and error log

# PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:/home/luca.palmieri/code/RL_puzzle_solver

dataset='images_with_50_lines'
pieces=8

# CONDA ENVIRONMENT
#conda activate repair
for puzzle in 'image_16' 'image_17' 'image_18' 'image_19' 'image_20' 
do
    echo "\nIMAGE $puzzle"
    echo "compatibility.."
    python compatibility/line_matching_NEW_segments.py --dataset $dataset --puzzle $puzzle --pieces $pieces #--penalty $3

    echo "solver.."
    for anchor in {0..64}
    do 
        echo "\nsolver with anchor $anchor"
        python solver/solverRotPuzzArgs.py --dataset $dataset --puzzle $puzzle --pieces $pieces --anchor $anchor --verbosity 0

        echo "\nevaluating with anchor $anchor"
        python metrics/evaluate.py --dataset $dataset --puzzle $puzzle --num_pieces $pieces --anchor $anchor

    done
done
