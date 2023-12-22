#!/bin/bash

#SBATCH --job-name=comp40                                             # Job name
#SBATCH --ntasks=40                                                      # Run on a single CPU
#SBATCH --output=RL_logs/irregular/compatibility40.log                         # Standard output and error log

# PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:/home/luca.palmieri/code/RL_puzzle_solver


echo "4pieces"
echo "## compatibility matrices.."
python compatibility/comp_irregular.py --dataset synthetic_irregular_4_pieces_by_drawing_lines_ccayvh --jobs 40 --method exact 

echo "16pieces"
echo "## compatibility matrices.."
python compatibility/comp_irregular.py --dataset synthetic_irregular_16_pieces_by_drawing_lines_ruyuvx --jobs 40 --method exact 

echo "real data"
echo "## compatibility matrices.."
python compatibility/comp_irregular.py --dataset synthetic_irregular_pieces_from_real_small_dataset --jobs 40 --method deeplsd 

