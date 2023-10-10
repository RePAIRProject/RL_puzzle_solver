#!/bin/bash/
echo "solving $2 from dataset $1 (with $3 pieces)"

echo "compatibility.."
python compatibility/line_matching_NEW_segments.py --dataset $1 --puzzle $2 --pieces $3 #--penalty $3

echo "solver.."
python solver/solverRotPuzzArgs.py --dataset $1 --puzzle $2 --pieces $3 --anchor $4
#--penalty $3

echo "evaluating.."
python metrics/evaluate.py --dataset $1 --puzzle $2 --num_pieces $3 

