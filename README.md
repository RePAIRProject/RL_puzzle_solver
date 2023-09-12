# RL Puzzle Solver
Solving puzzle using Relaxation Labeling

# 1) Description
The repository contains the code for computing features, the compatibility matrices and the solver.

## Preparing Data (`Preprocessing` folder)
Initially, we assume you have the input data for some puzzle (one image or the fragments).
The `preprocess.py` script creates the `data` subfolder for a certain puzzle. Once you have that, you can start computing stuff (computed stuff will go into the `output` folder).

## Compute Features (`features` folder)
You can use `compute_region_masks.py` to precompute candidate regions where the compatibility will be calculated, or `segment_with_yolov8.py` to create the segmentation masks or `lines_detection.py` to extract lines from the segmentation masks obtained. 

## File Naming Conventions
The folder structure is described below and the name are written down in the configuration file in the `configs` folder. Different configs can be used for different puzzle. The idea behind this is that given only a puzzle name, you can access via the config parameters the folders and read/write files using always the same convention.
## Folder Structure
The data is structured following this idea (where `data` has the input files and `output` all the files created by the code):
```bash
.
├── data/
│   ├── puzzle_1/
│   │   ├── images/
│   │   └── masks/
│   ├── puzzle_2/
│   │   ├── images/
│   │   └── masks/
└── output/
    ├── puzzle_1/
    │   ├── MotifSegmentation/
    │   ├── LinesDetection/
    │   ├── RegionsMatrix/
    │   └── CompatibilityMatrix/
    └── puzzle_2/
        ├── MotifSegmentation/
        ├── LinesDetection/
        ├── RegionsMatrix/
        └── CompatibilityMatrix/
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
```
Plus, if you need particular stuff:
```
YOLO-based segmentation: ultralytics
SDF-based compatibility: scikit-fmm
```

# 3) Usage
We use the `argparse` package for the parameters, so usually you pass parameters via command line and by using `-h` you can get some help on the parameters.

For example, to prepare the data:
```bash
python Preprocessing/preprocess.py -d path_to_data
```

To compute the regions pass the puzzle name (the name of the folder, `puzzle_1` or `puzzle_2` in the example folder structure above):
```bash
python features/compute_regions_masks.py --puzzle puzzle_name
```

Same for the compatibility:
```bash
python Compatibility/shape_compatibility.py --urm --puzzle puzzle_name
```
Here, you can add `--urm` if you use the regions matrix (computed using the script) to speed up calculations. Otherwise, remove `--urm` to calculate fully the matrix (much slower).

# 4) Known Issues
Bug descriptions.

# 5) Relevant publications
Some publications.

