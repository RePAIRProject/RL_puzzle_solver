#!/bin/bash/
echo "###############################################################"
echo "#                          PARAMETERS                         #"
echo "# ----------------------------------------------------------- #"
echo "#                            PUZZLE                           #"
echo "# dataset:    \t$1"
echo "# puzzle:     \t$2"
echo "# ----------------------------------------------------------- #"
echo "#                         COMPATIBILITY                       #"
echo "# method:         \t$3"
echo "# xy_step:        \t$4"
echo "# xy_grid_points: \t$5"
echo "# theta_step:     \t$6"
echo "# cmp_cost:       \t$7"
echo "# verbosity:      \t$8"
echo "# jobs:           \t$9"
echo "# ----------------------------------------------------------- #"
echo "#                            SOLVER                           #"
echo "# anchor:         \t$10"
echo "# tfirst:         \t$11"
echo "# tnext:          \t$12"
echo "# tmax:           \t$13"
echo "# p_pts:          \t$14"
echo "###############################################################"
echo ""

echo "REGION MASKS"
python features/compute_both_regions_masks.py --dataset $1 --puzzle $2 --method $3 --xy_step $4 --xy_grid_points $5 --theta_step $6

echo "COMPATIBILITY"
python compatibility/comp_irregular.py --dataset $1 --puzzle $2 --det_method $3 --cmp_cost $7 --verbosity $8 --jobs $9

echo "SOLVER"
python solver/solver_irregular.py --dataset $1 --puzzle $2 --det_method $3 --cmp_cost $7 --anchor $10 --tfirst $11 --tnext $12 --tmax $13 --p_pts $14

echo "FINISHED"