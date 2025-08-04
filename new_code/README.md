# Solving Jigsaw Puzzles 

Here we have the new version of the (cleaned) code for solving jigsaw puzzles using pairwise compatibilities and relaxation-labeling as a solver.

##### Idea
For details about the idea, please refer to the [paper](https://openaccess.thecvf.com/content/ACCV2024/papers/Khoroshiltseva_Nash_Meets_Wertheimer_Using_Good_Continuation_in_Jigsaw_Puzzles_ACCV_2024_paper.pdf).

##### Puzzle Generation
For the generation of the puzzles, check the other [repo](https://github.com/CVML-CFU/jigsaw_puzzle_toolkit)

## Code Workflow

0. Prepare the `input_parameters.yaml` file with all the necessary information (create from the `default.yaml` provided)
1. Generating the grid and the partial payoff matrix (Sec 4.1 of the paper), what we call *region matrix* (use `region_matrix.py`)
2. Calculating the payoff scores (Sec 4.2 of the paper, now also with different features, not only lines), what we call *compatibility matrix* (use `compatibility_matrix.py`)
3. Solving the puzzle based on the CM (Sec 3.3 of the paper), what we call *solver* (use `solve.py`)

It should create folders with the data (saved as `npy` dictionaries) and visualization.

More detailed explanations will follow. 