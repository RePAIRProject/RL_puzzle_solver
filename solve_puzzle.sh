#!/bin/bash/
echo "solving $2 from dataset $1 (with $3 pieces)"

echo "compatibility.."
python compatibility/line-matching_compatibility_2.py --dataset $1 --puzzle $2 #--penalty $3

echo "solver.."
python solver/solverRotPuzzArgs.py --dataset $1 --puzzle $2 --pieces $3 #--penalty $3

echo "evaluating.."
python metrics/evaluate.py --dataset $1 --puzzle $2 --num_pieces $3

