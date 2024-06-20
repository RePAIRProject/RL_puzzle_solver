| :exclamation:  This section is not updated!  |
|-----------------------------------------|


## Full pipeline from a dataset (folder with images)

Let's assume we have our dataset folder called `real_small_dataset`.
The input folder should be in the code folder + `data`
So full path could be: `~whatever_your_path~/RL_puzzle_solver/data/real_small_dataset`.

The output (everything we create) would be in the code folder + `output`.

### 1. Create pieces from images (fast, few seconds per image)
| :exclamation:  This section is not updated!  |
|-----------------------------------------|

Let's run the piece creation! It cuts our images into a (variable) number of pieces. 
We set the (maximum) number with the `-np` argument. It could lead to a smaller number of pieces, depending on the size of the image! This does not affect our algorithm, which does not strictly require a fixed number of pieces.

```bash
python datasets/create_pieces_from_images.py -i real_small_dataset -np 16 
```






### 2. Detect lines (fast, few seconds per piece, so less than a minute per image)
| :exclamation:  This section is not updated!  |
|-----------------------------------------|

The detection can be done with any edge detector. We sugget to use [DeepLSD](https://github.com/cvg/DeepLSD).
From the detected lines, we extract and save the initial and end points plus their polar coordinates (we will use the angle).
This script is actually launched from within the DeepLSD folder (for an easier usage of that) so it contains some hardcoded paths. 
You can define and change your own as needed.
```bash 
python detect_lines_irregular.py -rf ~whatever_your_path~/RL_puzzle_solver -d synthetic_irregular_pieces_from_real_small_dataset
```
The lines detected will be saved inside each folder of the database (there will be one `lines_detection` folder).
It also saves a visualization (image with lines drawn in red over it) and one representation with all white images with black lines drawn on top (without the real image colors).








| :exclamation:  These commands may be slightly outdated! Some things may change!  |
|-----------------------------------------|

<details>
<summary>See at your own risk!</summary>

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