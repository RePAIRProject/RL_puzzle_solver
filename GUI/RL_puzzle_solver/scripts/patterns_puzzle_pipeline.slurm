#!/bin/bash

#SBATCH --job-name=real_pattern
#SBATCH --ntasks=1                 				        # Run on a single CPU
#SBATCH --cpus-per-task=1
#SBATCH --output=RL_logs/pattern/real_pattern_%j.log			# Standard output and error log

# PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:/home/luca.palmieri/code/RL_puzzle_solver

dataset='maps_puzzle_patterns_10pcs_pieces_gkbvur_28_02_2024'
puzzle=$1

xy_step=30
xy_grid_points=9   ##  91  81 71
theta_step=90

det_method='exact'
cmp_cost_1='LAP'
cmp_cost_2='LCI'
verbosity=1
jobs=0

p_pts=31      #old#  125   151    175
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
echo "# cmp_cost:       \t$cmp_cost"
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
echo ""


#ech0 "GENERATE IMAGE"
#python data_generator/synth_puzzle.py -extr -s irregular -sv -np 9 -ni 10

#echo "REGION MASKS"
python features/compute_both_regions_masks.py --dataset $dataset --puzzle $puzzle --method $det_method --xy_step $xy_step --xy_grid_points $xy_grid_points --theta_step $theta_step

#echo "COMPATIBILITY"
python compatibility/comp_irregular.py --dataset $dataset --puzzle $puzzle --det_method $det_method --cmp_cost $cmp_cost_1 --verbosity $verbosity --jobs $jobs
#python compatibility/comp_irregular.py --dataset $dataset --puzzle $puzzle --det_method $det_method --cmp_cost $cmp_cost_2 --verbosity $verbosity --jobs $jobs

#echo "SOLVER"
python solver/solver_irregular.py --dataset $dataset --puzzle $puzzle --det_method $det_method --cmp_cost $cmp_cost_1 --anchor $anchor --tfirst $tfirst --tnext $tnext --tmax $tmax --p_pts $p_pts
#python solver/solver_irregular.py --dataset $dataset --puzzle $puzzle --det_method $det_method --cmp_cost $cmp_cost_2 --anchor $anchor --tfirst $tfirst --tnext $tnext --tmax $tmax --p_pts $p_pts


echo "FINISHED"
