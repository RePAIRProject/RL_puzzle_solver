#!/bin/bash

dataset='images_with_50_lines'
puzzle='image_21'
pieces=8

# CONDA ENVIRONMENT
#conda activate repair

#echo "compatibility.."
#python compatibility/line_matching_NEW_segments.py --dataset $dataset --puzzle $puzzle --pieces $pieces #--penalty $3

echo "solver.."
for (( anchor=0; anchor<=63; anchor++ ))
do 
    echo "solver with anchor $anchor"
    python solver/solverRotPuzzArgs.py --dataset $dataset --puzzle $puzzle --pieces $pieces --anchor $anchor --verbosity 0

    echo "evaluating with anchor $anchor"
    python metrics/evaluate.py --dataset $dataset --puzzle $puzzle --num_pieces $pieces --anchor $anchor

done
