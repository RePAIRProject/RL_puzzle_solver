#!/bin/bash

#SBATCH --job-name=real_lines_irr9pcs
#SBATCH --ntasks=1                 				        # Run on a single CPU
#SBATCH --cpus-per-task=2
#SBATCH --output=RL_logs/26_02/real_irr9pcs_%j.log			# Standard output and error log

# PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:/home/luca.palmieri/code/RL_puzzle_solver

dataset='maps_puzzle_square_64_pieces_yzqmpg_01_03'
puzzle=$1

xy_step=32
xy_grid_points=3   ##  91  81 71
theta_step=0

det_method='exact'
cmp_cost_1='LAP'
cmp_cost_2='LCI'
verbosity=2
jobs=24

p_pts=15      #old#  125   151    175
anchor=-1
tfirst=2000    #  1500  3000   5000
tnext=500      #  500   750    1000
tmax=9000      #  9000  15000  20000  


#!/bin/bash/
echo "###############################################################"
echo "#                          PARAMETERS                         #"
echo "# ----------------------------------------------------------- #"
echo "#                            PUZZLE                           #"
echo "# dataset:    \t$irr_puzzle_pipeline.slurm"
echo "# puzzle:     \t$puzzle"
echo "# ----------------------------------------------------------- #"
echo "#                         COMPATIBILITY                       #"
echo "# method:         \t$det_method"
echo "# xy_step:        \t$xy_step"
echo "# xy_grid_points: \t$xy_grid_points"
echo "# theta_step:     \t$theta_step"
echo "# cmp_cost:       \t$cmp_cost_1"
echo "# verbosity:      \t$verbosity"
echo "# jobs:           \t$jobs"
echo "# ----------------------------------------------------------- #"
echo "#                            SOLVER                           #"
echo "# anchor:         \t$anchor"
echo "# tfirst:         \t$tfirst"
echo "# tnext:          \t$tnext"
echo "# tmax:           \t$tmax"
echo "# p_pts:          \t$p_pts"
echo "###############################################################"


#ech0 "GENERATE IMAGE"
#python data_generator/synth_puzzle.py -extr -s irregular -sv -np 9 -ni 10


echo "-----------------------------------------------"
echo "REGION MASKS"
echo "python features/compute_square_regions_masks.py --dataset $dataset --puzzle $puzzle --method $det_method"
python features/compute_square_regions_masks.py --dataset $dataset --puzzle $puzzle --method $det_method

echo "-----------------------------------------------"
echo "COMPATIBILITY"
echo "python compatibility/comp_irregular.py --dataset $dataset --puzzle $puzzle --det_method $det_method --cmp_cost $cmp_cost_1 --verbosity $verbosity --jobs $jobs"
python compatibility/comp_irregular.py --dataset $dataset --puzzle $puzzle --det_method $det_method --cmp_cost $cmp_cost_1 --verbosity $verbosity --jobs $jobs

#python compatibility/comp_irregular.py --dataset $dataset --puzzle $puzzle --det_method $det_method --cmp_cost $cmp_cost_2 --verbosity $verbosity --jobs $jobs
echo "-----------------------------------------------"
echo "SOLVER"
echo "python solver/solver_irregular.py --dataset $dataset --puzzle $puzzle --det_method $det_method --cmp_cost $cmp_cost_1 --anchor $anchor --tfirst $tfirst --tnext $tnext --tmax $tmax --p_pts $p_pts"
python solver/solver_irregular.py --dataset $dataset --puzzle $puzzle --det_method $det_method --cmp_cost $cmp_cost_1 --anchor $anchor --tfirst $tfirst --tnext $tnext --tmax $tmax --p_pts $p_pts

#python solver/solver_irregular.py --dataset $dataset --puzzle $puzzle --det_method $det_method --cmp_cost $cmp_cost_2 --anchor $anchor --tfirst $tfirst --tnext $tnext --tmax $tmax --p_pts $p_pts


echo "FINISHED"
