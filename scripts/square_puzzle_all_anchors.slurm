#!/bin/bash

#SBATCH --job-name=square_maps
#SBATCH --ntasks=1                 				        # Run on a single CPU
#SBATCH --cpus-per-task=2
#SBATCH --output=RL_logs/square_03_03/square_maps_%j.log			# Standard output and error log

# PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:/home/luca.palmieri/code/RL_puzzle_solver

dataset='maps_puzzle_square_64_pieces_yzqmpg_01_03'
puzzle=$1

xy_step=32
xy_grid_points=3   ##  91  81 71
theta_step=0

det_method='exact'
cmp_cost='LAP'
verbosity=1
jobs=2

p_pts=15      #old#  125   151    175
tfirst=2000    #  1500  3000   5000
tnext=500      #  500   750    1000
tmax=6500      #  9000  15000  20000  


#!/bin/bash/
echo "###############################################################"
echo "#                          PARAMETERS                         #"
echo "# ----------------------------------------------------------- #"
echo "#                            PUZZLE                           #"
echo "# dataset:    \t$dataset"
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
echo "# anchor:         \tall"
echo "# tfirst:         \t$tfirst"
echo "# tnext:          \t$tnext"
echo "# tmax:           \t$tmax"
echo "# p_pts:          \t$p_pts"
echo "###############################################################"


echo "\n\n"

echo "-----------------------------------------------"
echo "SOLVER"

for anchor in {0..63}
    do 
        echo "solver with anchor $anchor"
        python solver/solver_irregular.py --dataset $dataset --puzzle $puzzle --det_method $det_method --cmp_cost $cmp_cost --anchor $anchor --tfirst $tfirst --tnext $tnext --tmax $tmax --p_pts $p_pts

    done

echo "-----------------------------------------------"
echo "EVALUATION"

python metrics/evaluate_squared_pieces.py -d $dataset -aa -p $puzzle

echo "-----------------------------------------------"
echo "FINISHED"