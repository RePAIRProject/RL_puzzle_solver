#!/bin/bash

#SBATCH --job-name=select_anchors		   		                        # Job name
#SBATCH --ntasks=1                 				                        # Run on a single CPU
#SBATCH --cpus-per-task=1
#SBATCH --output=RL_logs/anchors_selection.log			# Standard output and error log

# PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:/home/luca.palmieri/code/RL_puzzle_solver

from=$1
to=$2

# CONDA ENVIRONMENT
#conda activate repair
for puzzle_num in {0..30}
do
    echo "IMAGE_$puzzle_num"
    cp /home/m.khoroshiltseva/RL_puzzle_solver/output_8x8/random_lines_exact_detection/image_$puzzle_num/selected_anchor.json output_8x8/random_lines_exact_detection/image_$puzzl_num/.
done
