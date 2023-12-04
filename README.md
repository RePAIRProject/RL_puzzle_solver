# RL Puzzle Solver (WIP)
Solving puzzle using Relaxation Labeling

| :exclamation:  This repository is under active development! Some things may change!  |
|-----------------------------------------|

# 1) Description
The repository contains the code for computing features, the compatibility matrices and the solver.

## Preparing Data (`preprocessing` folder)
Initially, we assume you have the input data for some puzzle (one image or the fragments).
The `preprocess.py` script creates the `data` subfolder for a certain puzzle. Once you have that, you can start computing stuff (computed stuff will go into the `output` folder).

## Compute Features (`features` folder)
You can use `compute_region_masks.py` to precompute candidate regions where the compatibility will be calculated, or `segment_with_yolov8.py` to create the segmentation masks or `lines_detection.py` to extract lines from the segmentation masks obtained. 

## File Naming Conventions
The folder structure is described below and the name are written down in the `folder_names.py` file in the `configs` folder. 
sThe idea behind this is that given only a puzzle name, you can access via the config parameters the folders and read/write files using always the same convention.

## Parameters in the config files
Parameters are usually located in the `configs` folder. 
There are two main config files: `puzzle_from_fragments_cfg` and `puzzle_from_image_cfg` whose names should be self-explanatory. Inside the config we have the parameters for the size of the images and for the method to extract features.

## Folder Structure
The data is structured following this idea (where `data` has the input files and `output` all the files created by the code):
```bash
.
├── configs/        # configuration
│   ├── folder_names.py     # folders
│   └── puzzle_cfg.py       # parameters
├── preprocessing/      # preparing the puzzle
├── features/           # extracting features for the compatibility
├── compatibility/      # calculating the comp. matrices
├── solver/             # rl-based solver
├── data/               # not included here on Github
│   ├── wikiart/        # DATASET (collection of images)
│   │   ├── puzzle_1/   # PUZZLE (single image)
│   │   │   ├── images/ # color images
│   │   │   ├── pieces/ # fragments (created with the script)
│   │   │   └── masks/  # binary mask for the shape/contour (optional)
│   │   ├── puzzle_2/
│   │   │   ├── images/ # color images
│   │   │   ├── pieces/ # fragments (created with the script)
│   │   │   └── masks/  # binary mask for the shape/contour (optional)
└── output/             # only partially included
    ├── wikiart/        # DATASET name
        ├── puzzle_1/
        │   ├── motif_segmentation/      # motif segmentation (for repair)
        │   ├── lines_detection/         # detected lines
        │   ├── regions_matrix/          # candidate regions for comp speedup
        │   └── compatibility_matrix/    # final compatibility matrix 
        └── puzzle_2/
            ├── motif_segmentation/
            ├── lines_detection/
            ├── regions_matrix/
            └── compatibility_matrix/
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
scikit-learn
scikit-image
opencv-python-contrib
```
Plus, if you need particular stuff:
```
YOLO-based segmentation: ultralytics
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

## Full pipeline from a dataset (folder with images)

Let's assume we have our dataset folder called `real_small_dataset`.
The input folder should be in the code folder + `data`
So full path could be: `~whatever_your_path~/RL_puzzle_solver/data/real_small_dataset`.

The output (everything we create) would be in the code folder + `output`.

### 1. Create pieces from images (fast, few seconds per image)
Let's run the piece creation! It cuts our images into a (variable) number of pieces. 
We set the (maximum) number with the `-np` argument. It could lead to a smaller number of pieces, depending on the size of the image! This does not affect our algorithm, which does not strictly require a fixed number of pieces.

```bash
python datasets/create_pieces_from_images.py -i real_small_dataset -np 16 
```

The output of this would be saved in `~whatever_your_path~/RL_puzzle_solver/output/synthetic_irregular_pieces_from_real_small_dataset`
<details>
<summary>The output on the terminal should look something like: (Click to show)</summary>

```bash
######################################################################
#   Settings:
#  output : /home/lucap/code/RL_puzzle_solver/output
#  images : real_small_dataset
#  num_pieces : 16
#  shape : irregular
#  rescale : 1000
#  input_images : /home/lucap/code/RL_puzzle_solver/data/real_small_dataset
#  puzzle_folder : /home/lucap/code/RL_puzzle_solver/output/synthetic_irregular_pieces_from_real_small_dataset
######################################################################

--------------------------------------------------
image_00000_escher_day_and_night
	- done with piece 00000
	- done with piece 00001
	- done with piece 00002
	- done with piece 00003
	- done with piece 00004
	- done with piece 00005
	- done with piece 00006
	- done with piece 00007
	- done with piece 00008
	- done with piece 00009
	- done with piece 00010
	- done with piece 00011
	- done with piece 00012
	- done with piece 00013
	- done with piece 00014
Done with image_00000_escher_day_and_night: created 15 pieces.

```
And will continue with the next images..
</details>

### 2. Create region masks (rather slow, takes ~5 minutes per image)
After we have created our pieces, we create the regions masks.
```bash 
python features/compute_regions_masks.py --dataset synthetic_irregular_pieces_from_real_small_dataset --puzzle image_00000_escher_day_and_night
```
They will be created inside the puzzle folder (under `regions_matrix`).

**TIP:** if you remove the `--puzzle` argument, it will compute the regions for the whole dataset (this may take some time, usually some minutes (3 to 5) for each image).

<details>
<summary>The output on the terminal should look something like: (Click to show)</summary>

```bash
Found 15 pieces:
- piece_0000.png
- piece_0001.png
- piece_0002.png
- piece_0003.png
- piece_0004.png
- piece_0005.png
- piece_0006.png
- piece_0007.png
- piece_0008.png
- piece_0009.png
- piece_0010.png
- piece_0011.png
- piece_0012.png
- piece_0013.png
- piece_0014.png

##################################################
SETTINGS
The puzzle (maybe rescaled) has size 603x1000 pixels
Pieces are squared images of 268x268 pixels (p_hs=134)
The region matrix has shape: [101, 101, 24, 15, 15]
Using a grid on xy and 24 rotations on 15 pieces
	xy_step: 5.37, rot_step: 15.0
Canvas size: 537x537
##################################################

regions for pieces 14 and 14
Done calculating
##################################################
Saving the matrix..
Creating visualization
```
</details>

### 3. Detect lines (fast, few seconds per piece, so less than a minute per image)
The detection can be done with any edge detector. We sugget to use [DeepLSD](https://github.com/cvg/DeepLSD).
From the detected lines, we extract and save the initial and end points plus their polar coordinates (we will use the angle).
This script is actually launched from within the DeepLSD folder (for an easier usage of that) so it contains some hardcoded paths. 
You can define and change your own as needed.
```bash 
python detect_lines_irregular.py -rf ~whatever_your_path~/RL_puzzle_solver -d synthetic_irregular_pieces_from_real_small_dataset
```
The lines detected will be saved inside each folder of the database (there will be one `lines_detection` folder).
It also saves a visualization (image with lines drawn in red over it) and one representation with all white images with black lines drawn on top (without the real image colors).

### 4. Compute compatibility (slow, computer pairwise, few minutes per pair, so some hours per puzzle!)





| :exclamation:  These commands may be slightly outdated! Some things may change!  |
|-----------------------------------------|

<details>
<summary>(Click to show)</summary>

For example, to prepare the data from images:
```bash
python preprocessing/preprocess_image_dataset.py -d dataset_name
```

Or to prepare the data from fragments
```bash
python preprocessing/preprocess_fragments.py -d dataset_name
```

To compute the regions pass the puzzle name (the name of the folder, `puzzle_1` or `puzzle_2` in the example folder structure above):
```bash
python features/compute_regions_masks.py --puzzle puzzle_name
```

Same for the compatibility:
```bash
python compatibility/shape_compatibility.py --urm --puzzle puzzle_name
```
Here, you can add `--urm` if you use the regions matrix (computed using the script) to speed up calculations. Otherwise, remove `--urm` to calculate fully the matrix (much slower).




## Full pipeline (example)

#### Create pieces
```bash
python preprocessing/preprocess_image_dataset.py -d manual_lines
```

#### Detect lines (using deepLSD)
```bash
python detect_lines_compatibility.py -rf /home/lucap/code/RL_puzzle_solver -d manual_lines
```

#### Compute compatibility
```bash
python compatibility/line_matching_NEW_segments.py --dataset manual_lines --puzzle lines1
```

#### Solver
```bash
python solver/solverRotPuzzArgs.py --dataset manual_lines --puzzle lines1
```

#### Evaluation
```bash
python metrics/evaluate.py
```


## Full pipeline RePAIR

#### Create pieces ???
```bash
python preprocessing/preprocess_fragments.py -d repair
```

#### Detect lines (using deepLSD)
```bash
python detect_lines_compatibility.py -rf /home/lucap/code/RL_puzzle_solver -d repair
```

#### Compute regions matrices
```bash
python features/compute_regions_masks.py --dataset repair --puzzle decor_1
```

#### Compute polygons
```bash
python dataset/create_polygons.py 
```

#### Compute compatibility
```bash
python compatibility/line_matching_RePAIR_poly_2910.py --dataset repair --puzzle decor_1
```

## SCRIPTS to do multiple steps at once


#### Solve a puzzle (given pieces and detected lines)
```bash
sh solve_puzzle.sh
```
</details>

# 4) Known Issues

***Problem with input and output data folders path?***

Input and output (respectively `data_path = 'data'` and `output_dir = 'output'` in the config file) are defined without the full path. If you run the scripts via terminal from the root folder, you should be fine. If you run from subfolders or use special settings, you can set these two accordingly.

# 5) Relevant publications
Hopefully soon.

# 6) Acknowledgements
We use part of other open source software/tools:
- [PuzzleSolving-tool](https://github.com/xmlyqing00/PuzzleSolving-tool) from [Yongqing Liang ](https://github.com/xmlyqing00) // We used a modified version included in this repo under `puzzle_utils/puzzle_gen` from [our fork of their framework](https://github.com/RePAIRProject/2DPuzzleSolving-tool)
- [DeepLSD](https://github.com/cvg/DeepLSD) from [Computer Vision and Geometry Lab, ETH Zurich](https://github.com/cvg)

