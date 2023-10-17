#!/bin/bash

#SBATCH --job-name=eval_64anchors		   		# Job name
#SBATCH --ntasks=1                 				                    # Run on a single CPU
#SBATCH --cpus-per-task=1
#SBATCH --output=RL_logs/64anchors_50lines_evaluation_img%j.log			    # Standard output and error log

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
    echo "IMAGE_$puzzle_num"
    echo "evaluating for each anchor"
    python metrics/evaluate.py --dataset $dataset --puzzle image_$puzzle_num --num_pieces $pieces --all_anchors

    echo "anchor analysis"
    python metrics/evaluate_multiple_anchors.py --puzzle image_$puzzle_num

done
