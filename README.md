# RL Puzzle Solver (WIP)
Solving puzzle using Relaxation Labeling

# 1) Description
The repository contains the code for computing the compatibility matrices and the solver to reassemble the pieces.


## Folder Structure
The data is structured following this idea (where `data` has the input files and `output` all the files created by the code):
```bash
.
├── configs/                        # configuration
│   ├── folder_names.py             # folder names to use it systematically
├── compatibility/                  # calculating the comp. matrices
├── data_generator/                 # code for creating puzzles (synthetic and real)
├── datasets/                       # (old) code for lines creation
├── features/                       # extracting features for the compatibility
├── metrics/                        # evaluating the solutions
├── preprocessing/                  # needed for special cases to prepare the images of the pieces
├── puzzle_utils/                   # under the hood core functions
│   ├── puzzle_gen/                 # generating pieces from image (derived from puzzle solving tools (ref below))
│   └── various_files.py            # containing functions for handling pieces and polygon,
│                                   # creating maps, lines matching and more
├── scripts/                        # bash script to launch several execution (mostly on the server using slurm)
├── solver/                         # RL-based solver
└── output/                         # datasets will be generated here  
    ├── dataset1/                   # DATASET name
    │   ├── image_0000/             # image (is a folder with all you need)
    │   ├── image_0001/
    │   ├── ...                     # the number varies
    │   └── parameters.json         # parameters about the creation (if created with our code)
    └── dataset2/                   # same, more datasets can be placed there
        ├── image_0000/
        ├── image_0001/
        └── parameters.json
```

# 2) Installation
Depending on the features used, you may need to install different libraries. 
We rely heavily on:
```
numpy
opencv-python
matplotlib
argparse
```
We use slightly less but still most likely to be needed:
```
scipy
skfmm
scikit-learn
scikit-image
opencv-python-contrib
```
Plus, if you need particular stuff:
```
YOLO-based segmentation/OBB: ultralytics
SDF-based compatibility: scikit-fmm
DeepLSD for line detection: [DeepLSD](https://github.com/cvg/DeepLSD)
```

# 3) Usage
We use the `argparse` package for the parameters, so usually you pass parameters via command line and by using `-h` you can get some help on the parameters.

The steps needed are:

- Optional: Create the pieces from an image
- Create regions mask to filter out candidate positions
- Compute compatibility matrix
- Launch the solver to reach the puzzle solution
- Evaluate results


## Pipeline used for ECCV 2024

### 1. Create synthetic line-based puzzles

We have a `synth_puzzle.py` script which controls the creation of the puzzles, allowing you to tweak:
- number of pieces (`-np`)
- number of lines on the puzzle/image (`-nl`)
- number of puzzle/images to create (`-ni`)
- many more.. (run `python data_generator/synth_puzzle.py -h` for list of arguments!)

Examples:

#### Ten images with 30 lines each cut into irregular pieces
```bash
python data_generator/synth_puzzle.py -nl 30 -sv -ni 10 -s irregular
```

#### One image with 30 lines cut using a pattern found in `data/patterns` (with extrapolated version)
This requires patterns in a folder (which in the below command is assumed to be `data/patterns`)
```bash
python data_generator/synth_puzzle.py -nl 30 -sv -ni 1 -s pattern -pf data/patterns --extr
```

##### Arguments
Arguments are the same: `-nl` is the number of lines, `-sv` saves the lines visualization, `-ni` is the number of images, `-s` the shape, `-extr` extrapolates the fragments, `-pf` is the pattern folder.

You can get the full list of arguments options running 
```bash
python data_generator/synth_puzzle.py -h
```

The output of this will be saved in `~whatever_your_path~/RL_puzzle_solver/output/synthetic_irregular_or_patterns+random_string`

### 2. Compatibility Matrix $R$

We calculate scores for all possible pairwise combinations on our compatibility matrix `R`. 
The compatibility matrix has 5 dimensions. 
One compatibility score can be written as $R(x, y, \theta, i, j) = s$. The compatibility score $s$ is the score when we place the piece $i$ in the center $x_i=0, y_i=0, \theta_i=0$ and we place the piece $j$ at $x, y, \theta$! 
So the $x,y,\theta$ refer to piece $j$ and not to piece $i$ (which is fixed in the center).
The compatibility matrix is pairwise (so it contains information about 2 pieces) and relative (piece $i$ against piece $j$). Please refer to the paper for more info. 

### 2a. Create region masks 

Since it is clear that many combinations are useless (if pieces overlap or are far away from each other) we can estimate a much smaller region where we are interested to calculate the compatibility and use it to reduce computations. We call this the "region" mask, where it creates a pixel-wise map with `1` values where we are interested in, `0` values on areas where pieces are far away from each other and `-1` if the two pieces overlap.
Depending on which pieces you have, you may get very different results in terms of performances.
This highly depends on the shape of the (pairwise) compatbility matrix (the region matrix has the same shape)

We can compute the regions mask with the command:
```bash 
python features/compute_both_regions_masks.py --dataset dataset_name --puzzle image_00000 --method exact --xy_step 30 --xy_grid_points 7 --theta_step 90
```
Where `--dataset` selects the dataset, `--puzzle` the image/puzzle, `--method` how the lines were extracted (usually `exact` or `deeplsd`), `-xy_step` is the distance between two possible candidate position on the `xy` plane, `--theta_step` the same in the rotation space (in degrees).
The `--xy_grid_points` control the shape of the grid (which, if you set it to $N$, will be $N \times N$)

The script will create and save the outcome inside the puzzle folder (under `regions_matrix`).

**TIP:** if you remove the `--puzzle` argument, it will compute the regions for the whole dataset.

You can get the full list of arguments options running 
```bash
python features/compute_both_regions_masks.py -h 
```

### 2b. Compute compatibility (may take some time on irregular pieces)

The compatibility can be computed using:
```bash
python compatibility/comp_irregular.py --dataset dataset_name --puzzle image_name --det_method exact --cmp_cost LAP --xy_step 30 --xy_grid_points 7 --theta_step 90 --verbosity 1
```
Where `--dataset` selects the dataset, `--puzzle` the image/puzzle,  `--jobs` can be used to run in parallel the computations, `--det_method` is the method used to extract lines (`deeplsd` or `exact` for example), 
`cmp_cost` chooses the algorithm to compute the compatibility cost (at the moment we have implemented `LAP` and `LCI`, see below for more details), `--penalty` is the penalty value (to use the correct compatibility matrix),
the `_step`, `_grid_points` are inherited from the region matrix (should be the same), `-verbosity` controls how much of what is happening is printed (to screen or log).

You can get the full list of arguments options running 
```bash
python compatibility/comp_irregular.py -h 
```

#### Compatibility Cost Algorithms
If you are wondering about `-cmp_cost` argument, we have in the code two possible implementations

##### Linear Assignment Problem (LAP)
Given two pieces with features, it tries to find a correspondence/matching between them. It does so by solving the related [linear assimgnent problem](https://en.wikipedia.org/wiki/Assignment_problem#Solution_by_linear_programming) and assigns as a score the total costs of the matching. It is very good when the number of features on the two sets is the same (in this case, number of lines in the two pieces), and we introduced a penalty when this does not hold and there remains unmatched features (lines)

##### Line Confidence Importance (LCI)
This approach is more specific to the lines, and it uses a positive-negative contribution strategy. Meaning for each line, it adds a positive contribution for each line which is "continued" on the other piece (the contribution is the multiplication between confidence of detection and importance of the feature) and a negative contribution for each line which is not "continued".

### 5. Running the solver to get the solution (slow, half an hour per puzzle)

The solver takes as input the compatibility matrix and starts from there. 
We tune the solver by selecting the `anchor` and the period `T`.  

The solver can be launched with
```bash
python solver/solver_irregular.py --dataset dname --puzzle image_00005 --anchor 5 --Tfirst 1000 --Tnext 500 --Tmax 3000
```
Where in this case we select the 5-th piece as an anchor and we run `Tfirst=1000` iterations before stopping, checking and fixing the other anchors. Then the process is repeated each `Tnext=500` iterations until reaching `Tmax=3000` and stopping.

### 6. Evaluating the results

Given you have ground truth (for synthetic case or when you have the solution), you can evaluate the reconstruction.
The code changes if you are using irregular or squared pieces (we have different metrics).

An example for squared pieces is:
```bash
python metrics/evaluate_squared_pieces.py -d maps_dataset -aa -p image_00003_barcelona_catalonia_es
```
In this case `-aa` means *evaluate for all anchors* (useful if you tried several anchors for evaluation)
An example for irregular pieces is:   
```bash
python metrics/evaluate_irregular_new.py -d maps_pattern_or_irregular -p image_00001
```

### (Extra) Script for full pipeline (given you have a puzzle in a dataset)
```bash
sh scripts/synth_puzzle_pipeline.sh synthetic_irregular_9_pieces_by_drawing_coloured_lines_jqdmhs image_00000 exact 10 31 90 LCI 1 24 5 600 300 1500 111
```
This may be very hard to read, but under the hood just runs the same command from step 3 to 5 passing the parameters (open the file for more info)


## Experiments

We have much more code for experiments (specific ones) and old versions, it is (more or less) documented in [experiments.md](https://github.com/RePAIRProject/RL_puzzle_solver/blob/develop/experiments.md), use it if you are curious.



# 4) Known Issues

***Problem with input and output data folders path?***

Input and output path, respectively `data_path = 'data'` and `output_dir = 'output'` in the config file (`configs/folder_names.py`) are defined without the full path. If you run the scripts via terminal from the root folder, you should be fine. If you run from subfolders or use special settings, you can set these two accordingly.

***Issues when importing `shapely`***

If you get `ImportError: cannot import name 'transform' from 'shapely' (/home/lucap/miniconda3/lib/python3.9/site-packages/shapely/__init__.py)` then you probably have an old version of shapely. Updating it (I guess `pip install --update shapely` should be enough) usually solves the problem.

# 5) Relevant publications
Hopefully soon.

# 6) Acknowledgements
We use part of other open source software/tools:
- [PuzzleSolving-tool](https://github.com/xmlyqing00/PuzzleSolving-tool) from [Yongqing Liang ](https://github.com/xmlyqing00) // We used a modified version included in this repo under `puzzle_utils/puzzle_gen` from [our fork of their framework](https://github.com/RePAIRProject/2DPuzzleSolving-tool)
- We recommend using [DeepLSD](https://github.com/cvg/DeepLSD) from [Computer Vision and Geometry Lab, ETH Zurich](https://github.com/cvg) for detecting lines on real images.

