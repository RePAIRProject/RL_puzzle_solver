#!/bin/bash

#SBATCH --job-name=a64i5-10_50lines_puzzle		   		# Job name
#SBATCH --ntasks=5                 				        # Run on a single CPU
#SBATCH --cpus-per-task=5
#SBATCH --output=RL_logs/64anchors_50lines_img5to10.log			    # Standard output and error log

# PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:/home/luca.palmieri/code/RL_puzzle_solver

dataset='images_with_50_lines'
pieces=8

# CONDA ENVIRONMENT
#conda activate repair
for image in 'image_6' 'image_7' 'image_8' 'image_9' 'image_10'
do
    echo "\nIMAGE $image"
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
done
