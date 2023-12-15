#!/bin/bash

#SBATCH --job-name=irregulars                                  # Job name
#SBATCH --ntasks=1                                                      # Run on a single CPU
#SBATCH --output=RL_logs/irregular/reg_comp_dic15.log                         # Standard output and error log

# PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:/home/luca.palmieri/code/RL_puzzle_solver


echo "4pieces"
echo "## region matrices.."
#echo "python features/compute_both_regions_masks.py --dataset $dataset --puzzle $puzzle --pieces $pieces --method $method"
python features/compute_both_regions_masks.py --dataset synthetic_irregular_4_pieces_by_drawing_lines_ccayvh --method exact

echo "## compatibility.."
#echo "python compatibility/comp_irregular.py --dataset $dataset --puzzle $puzzle --pieces $pieces --method $method"
python compatibility/comp_irregular.py --dataset synthetic_irregular_4_pieces_by_drawing_lines_ccayvh --jobs 0 --method exact 


echo "16pieces"
echo "## region matrices.."
#echo "python features/compute_both_regions_masks.py --dataset $dataset --puzzle $puzzle --pieces $pieces --method $method"
python features/compute_both_regions_masks.py --dataset synthetic_irregular_16_pieces_by_drawing_lines_ruyuvx --method exact

echo "## compatibility.."
#echo "python compatibility/comp_irregular.py --dataset $dataset --puzzle $puzzle --pieces $pieces --method $method"
python compatibility/comp_irregular.py --dataset synthetic_irregular_16_pieces_by_drawing_lines_ruyuvx --jobs 0 --method exact 


echo "real data"
echo "## region matrices.."
#echo "python features/compute_both_regions_masks.py --dataset $dataset --puzzle $puzzle --pieces $pieces --method $method"
python features/compute_both_regions_masks.py --dataset synthetic_irregular_pieces_from_real_small_dataset --method exact

echo "## compatibility.."
#echo "python compatibility/comp_irregular.py --dataset $dataset --puzzle $puzzle --pieces $pieces --method $method"
python compatibility/comp_irregular.py --dataset synthetic_irregular_pieces_from_real_small_dataset --jobs 0 --method exact 



# echo "## solver.."
# echo "python solver/solverRotPuzzArgs.py --dataset $dataset --puzzle $puzzle --pieces $pieces --anchor $anchor"
# python solver/solverRotPuzzArgs.py --dataset $dataset --puzzle $puzzle --pieces $pieces --anchor $anchor

# echo "## evaluating.."
# echo "python metrics/evaluate.py --dataset $dataset --puzzle $puzzle --num_pieces $pieces "
# python metrics/evaluate.py --dataset $dataset --puzzle $puzzle --num_pieces $pieces 
