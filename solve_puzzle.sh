#!/bin/bash/
echo "solving $2 from dataset $1 (with $3 pieces)"

echo "compatibility.."
python compatibility/line-matching_compatibility_2.py --dataset $1 --puzzle $2 #--penalty $3

echo "solver.."
python solver/solverRotPuzzArgs.py --dataset $1 --puzzle $2 --pieces $3 #--penalty $3

echo "evaluating.."
python metrics/evaluate.py -s /home/lucap/code/RL_puzzle_solver/output_8x8/$1/$2/solution/p_final.mat --dataset $1 -pz $2 -o /home/lucap/code/RL_puzzle_solver/output_8x8/$1/$2/solution

