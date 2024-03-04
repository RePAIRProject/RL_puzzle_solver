#!/bin/bash

#SBATCH --job-name=eval_irregular
#SBATCH --ntasks=1  
#SBATCH --cpus-per-task=1
#SBATCH --output=RL_logs/evaluation_irregular_%j.log

# PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:/home/luca.palmieri/code/RL_puzzle_solver

#square_dataset_synth=''
pattern_dataset_synth='synthetic_patterns_10pcs_pieces_by_drawing_coloured_lines_rxbobr_26_02_2024'
irregular_dataset_synth='synthetic_irregular_9_pieces_by_drawing_coloured_lines_kxdwtu_26_02_2024'

#square_dataset_real='maps_puzzle_square_64_pieces_yzqmpg_01_03'
pattern_dataset_real='maps_puzzle_patterns_10pcs_pieces_gkbvur_28_02_2024'
irregular_dataset_real='maps_puzzle_irregular_9_pieces_rzwlyg_28_02_2024'

echo "-----------------------------------------------"
echo "SYNTHETIC"
echo "python metrics/write_neighbours_list.py --dataset $pattern_dataset_synth"
python metrics/write_neighbours_list.py --dataset $pattern_dataset_synth
echo "python metrics/evaluate_irregular_pieces.py --dataset $pattern_dataset_synth"
python metrics/evaluate_irregular_pieces.py --dataset $pattern_dataset_synth

echo "python metrics/write_neighbours_list.py --dataset $irregular_dataset_synth"
python metrics/write_neighbours_list.py --dataset $irregular_dataset_synth
echo "python metrics/evaluate_irregular_pieces.py --dataset $irregular_dataset_synth"
python metrics/evaluate_irregular_pieces.py --dataset $irregular_dataset_synth 

echo "-----------------------------------------------"
echo "REAL"
echo "python metrics/write_neighbours_list.py --dataset $pattern_dataset_real"
python metrics/write_neighbours_list.py --dataset $pattern_dataset_real
echo "python metrics/evaluate_irregular_pieces.py --dataset $pattern_dataset_real"
python metrics/evaluate_irregular_pieces.py --dataset $pattern_dataset_real

echo "python metrics/write_neighbours_list.py --dataset $irregular_dataset_real"
python metrics/write_neighbours_list.py --dataset $irregular_dataset_real
echo "python metrics/evaluate_irregular_pieces.py --dataset $irregular_dataset_real"
python metrics/evaluate_irregular_pieces.py --dataset $irregular_dataset_real 

echo "FINISHED"
