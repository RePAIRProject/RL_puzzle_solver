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
python compatibility/line_matching_segments.py --dataset manual_lines --puzzle lines1
```

#### Solver
```bash
python solver/solverRotPuzzArgs.py --dataset manual_lines --puzzle lines1
```


# 4) Known Issues

***Problem with input and output data folders path?***

Input and output (respectively `data_path = 'data'` and `output_dir = 'output'` in the config file) are defined without the full path. If you run the scripts via terminal from the root folder, you should be fine. If you run from subfolders or use special settings, you can set these two accordingly.

# 5) Relevant publications
Hopefully soon.

