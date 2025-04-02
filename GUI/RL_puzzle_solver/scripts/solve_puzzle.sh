#!/bin/bash/
echo "solving $2 from dataset $1 (with $3 pieces)"

dataset=$1
puzzle=$2
pieces=$3
method=$4
anchor=27

echo "## compatibility.."
echo "python compatibility/line_matching_NEW_segments.py --dataset $dataset --puzzle $puzzle --pieces $pieces --method $method"
python compatibility/line_matching_NEW_segments.py --dataset $dataset --puzzle $puzzle --pieces $pieces --method $method

echo "## solver.."
echo "python solver/solverRotPuzzArgs.py --dataset $dataset --puzzle $puzzle --pieces $pieces --anchor $anchor"
python solver/solverRotPuzzArgs.py --dataset $dataset --puzzle $puzzle --pieces $pieces --anchor $anchor

echo "## evaluating.."
echo "python metrics/evaluate.py --dataset $dataset --puzzle $puzzle --num_pieces $pieces "
python metrics/evaluate.py --dataset $dataset --puzzle $puzzle --num_pieces $pieces 
