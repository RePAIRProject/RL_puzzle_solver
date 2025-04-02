#!/bin/bash

#SBATCH --job-name=irregulars                                  # Job name
#SBATCH --ntasks=1                                                      # Run on a single CPU
#SBATCH --output=RL_logs/irregular/reg_comp_dic15.log                         # Standard output and error log

# PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:/home/luca.palmieri/code/RL_puzzle_solver


echo "4pieces"
echo "## region matrices.."
python features/compute_both_regions_masks.py --dataset synthetic_irregular_4_pieces_by_drawing_lines_ccayvh --method exact

echo "16pieces"
echo "## region matrices.."
python features/compute_both_regions_masks.py --dataset synthetic_irregular_16_pieces_by_drawing_lines_ruyuvx --method exact

echo "real data"
echo "## region matrices.."
python features/compute_both_regions_masks.py --dataset synthetic_irregular_pieces_from_real_small_dataset --method deeplsd

