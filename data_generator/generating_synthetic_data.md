# Generating synthetic data
To generate a set of images, run:

```
python data_generator/synth_puzzle.py -s pattern -pf data/patterns_10pcs -nl 3 -th 20 -ni 1 -sv
```

### Parameters
- `-s pattern` cuts the pieces following the image pattern
- `-pf data/patterns_10pcs` this is the _reference_ image for the cutting the pieces: you need to place the `pattern_10pcs` folder inside a `data` folder in the root of the code (so it should be: `~/whatever/RL_puzzle_solver/data/patterns_10pcs`).
- `-nl 3` means you draw 3 lines on the image (`-nl 50` would draw $50$ lines)
- `-th 20` is the thickness (in pixels) of the line (if you want something more like the _bands_)
- `-ni 1` will create 1 image (puzzle). More images will have different random lines but same patterns pieces.
- `-sv` means `save_visualization` which saves more visual related to the pieces/cutting.
The complete list of parameters can be seen running `data_generator/synth_puzzle.py -h` 

### Drawing lines
The code which creates the random lines in inside `RL_puzzle_solver/puzzle_utils/dataset_gen.py`, the method is called `create_random_image(line_type, num_lines, width, height, is_closed = False, thickness = 1, col = 0)`.
It uses inside the `cv2.line` function (so you can replace it with something for the curve drawing)