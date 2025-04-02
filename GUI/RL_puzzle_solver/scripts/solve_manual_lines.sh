#!/bin/bash/

# manual lines dataset

# compute compatibilities
#echo "lines1"
#python compatibility/line_matching_NEW_segments.py --dataset manual_lines --puzzle lines1 --pieces 8
echo "lines2"
python compatibility/line_matching_NEW_segments.py --dataset manual_lines --puzzle lines2 --pieces 8
echo "lines3"
python compatibility/line_matching_NEW_segments.py --dataset manual_lines --puzzle lines3 --pieces 8
echo "lines4"
python compatibility/line_matching_NEW_segments.py --dataset manual_lines --puzzle lines4 --pieces 8
echo "lines5"
python compatibility/line_matching_NEW_segments.py --dataset manual_lines --puzzle lines5 --pieces 8
# # echo "colors"
# python compatibility/line_matching_NEW_segments.py --dataset manual_lines --puzzle colors

# launch solver
#echo "lines1"
#python solver/solverRotPuzzArgs.py --dataset manual_lines --puzzle lines1 --pieces 8
echo "lines2 solver" 
python solver/solverRotPuzzArgs.py --dataset manual_lines --puzzle lines2 --pieces 8 
echo "lines3 solver"
python solver/solverRotPuzzArgs.py --dataset manual_lines --puzzle lines3 --pieces 8
echo "lines4 solver"
python solver/solverRotPuzzArgs.py --dataset manual_lines --puzzle lines4 --pieces 8
echo "lines5 solver" 
python solver/solverRotPuzzArgs.py --dataset manual_lines --puzzle lines5 --pieces 8
# echo "colors"
# python solver/solverRotPuzzArgs.py --dataset manual_lines --puzzle colors 


