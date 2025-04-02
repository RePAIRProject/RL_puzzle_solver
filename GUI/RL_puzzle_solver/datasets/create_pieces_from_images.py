import os, json, pdb, argparse
import shapely
import cv2
import numpy as np
import matplotlib.pyplot as plt
# import helper functions
from puzzle_utils.dataset_gen import generate_random_point, create_random_image
from puzzle_utils.lines_ops import line_cart2pol
from puzzle_utils.pieces_utils import cut_into_pieces, rescale_image

"""
This script generate new datasets for puzzle solving based on lines:

# GENERATION
It generates any number of images by:
- drawing random lines (segments, polylines, lines, ecc..) on it
- cutting the image into pieces

# OUTPUT
It creates one json file with the parameters and two folders:
- data (where the images will be stored)
- puzzle (where the pieces and their extracted lines are stored)

## Data
Inside `data` you find all the images,
with their name (image_00000.jpg, image_00001.jpg, and so on)

## Puzzle
Inside `puzzle` you find a folder for each image created.
Inside these folders (inside image_0000 for example) you find:
- pieces (a folder which contains all the pieces)
    pieces can be squared or irregular and they will be
    saved as piece_0000.png with transparent background
- lines_detection/exact 
    two folders, because you can later have other detection methods
    inside `exact` you have a piece_0000.json file with the line parameters


Example:
.
├── parameters.json                         # parameters used in the creation
├── data/                                   # the images created
│   ├── image_00000.jpg
│   ├── image_00001.jpg
│   ├── image_00002.jpg
|   ├── ..
│   └── image_NNNNN.jpg      
└── puzzle/               
    ├── image_00000/                        # output puzzle pieces     
    │   ├── regions_colorcoded.jpg   
    │   ├── regions_uint8.jpg   
    │   ├── pieces/   
    │   |   ├── piece_0000.png
    │   |   ├── piece_0001.png
    │   |   ├── piece_0002.png
    │   |   ├── ..
    │   |   └── piece_KKKK.png
    |   └── lines_detection/                # lines detection folder
    |       └── exact/                      # here the exact lines
    |           ├── piece_0000.json 
    |           ├── piece_0001.json
    |           ├── piece_0002.json
    |           ├── ..
    │           └── piece_KKKK.png
    ├── image_00001/
    ├── image_00002/
    ├── ..
    └── image_NNNNN/
"""

def main(args):

    if args.output == '':
        output_root_path = os.path.join(os.getcwd(), 'output')
    else:
        output_root_path = args.output
    
    if args.images.find('/') < 0:
        input_images_path = os.path.join(os.getcwd(), 'data', args.images)
        data_name = args.images
    else:
        input_images_path = args.images 
        data_name = input_images_path.split('/')[-1]

    dataset_name = f"synthetic_{args.shape}_pieces_from_{data_name}"
    puzzle_folder = os.path.join(output_root_path, dataset_name)
    parameter_path = os.path.join(puzzle_folder, 'parameters.json')
    os.makedirs(puzzle_folder, exist_ok=True)

    # save parameters
    parameters_dict = vars(args) # create dict from namespace
    parameters_dict['output'] = output_root_path
    parameters_dict['input_images'] = input_images_path
    parameters_dict['puzzle_folder'] = puzzle_folder
    print()
    print("#" * 70)
    print("#   Settings:")
    for parkey in parameters_dict.keys():
        print("# ", parkey, ":", parameters_dict[parkey])
    print("#" * 70)
    print()
    
    
    images_names = os.listdir(input_images_path)
    num_pieces_dict = {}
    img_sizes_dict = {}
    for k, img_path in enumerate(images_names):

        img_parameters = {}
        ## create images with lines
        img = cv2.imread(os.path.join(input_images_path, img_path))
        if max(img.shape[:2]) > args.rescale:
            img = rescale_image(img, args.rescale)

        ## save created image
        #cv2.imwrite(os.path.join(data_folder, f'image_{N:05d}.jpg'), img)

        ## make folders to save pieces and detected lines
        img_name = img_path[:-4]
        print("-" * 50)
        puzzle_name = f'image_{k:05d}_{img_name}'
        print(puzzle_name)
        single_image_folder = os.path.join(puzzle_folder, puzzle_name)
        scaled_image_folder = os.path.join(single_image_folder, 'image_scaled')
        os.makedirs(scaled_image_folder, exist_ok=True)
        cv2.imwrite(os.path.join(scaled_image_folder, f"output_{img_name}.png"), img)
        pieces_single_folder = os.path.join(single_image_folder, 'pieces')
        os.makedirs(pieces_single_folder, exist_ok=True)
        masks_single_folder = os.path.join(single_image_folder, 'masks')
        os.makedirs(masks_single_folder, exist_ok=True)
        poly_single_folder = os.path.join(single_image_folder, 'polygons')
        os.makedirs(poly_single_folder, exist_ok=True)
        single_image_parameters_path = os.path.join(single_image_folder, f'parameters_{puzzle_name}.json')

        # this gives us the pieces
        pieces, patch_size = cut_into_pieces(img, args.shape, args.num_pieces, single_image_folder, puzzle_name)
        # pieces is a list of dicts with several keys:
        # - pieces[i]['centered_image'] is the centered image 
        # - pieces[i]['centered_mask'] is the centered mask (as binary image)
        # - pieces[i]['centered_polygon'] is the shape (as a shapely polygon)
        # - pieces[i]['squared_image'] is the centered image in a square
        # - pieces[i]['squared_mask'] is the centered mask (as binary image) in a square
        # - pieces[i]['squared_polygon'] is the centered shape (as a shapely polygon) in the square
        # - pieces[i]['polygon'] is the shape (as a shapely polygon) in its original position
        # - pieces[i]['shift2center'] is the translation to align the shape to the center of the square

        for j, piece in enumerate(pieces):
            ## save patch
            cv2.imwrite(os.path.join(pieces_single_folder, f"piece_{j:04d}.png"), piece['squared_image'])
            cv2.imwrite(os.path.join(masks_single_folder, f"piece_{j:04d}.png"), piece['squared_mask'] * 255)
            np.save(os.path.join(poly_single_folder, f"piece_{j:04d}"), piece['squared_polygon'])
            print(f'\t- done with piece {j:05d}')

        num_pieces_dict[puzzle_name] = j+1
        img_sizes_dict[puzzle_name] = img.shape

        # parameters of the single image!
        img_parameters['piece_size'] = int(patch_size)
        img_parameters['num_pieces'] = j+1
        img_parameters['size'] = img.shape

        with open(single_image_parameters_path, 'w') as pp:
            json.dump(img_parameters, pp, indent=2)
        
        print(f"Done with {puzzle_name}: created {j+1} pieces.")

    # parameters for the whole dataset
    parameters_dict['num_pieces'] = num_pieces_dict
    parameters_dict['img_sizes'] = img_sizes_dict
    with open(parameter_path, 'w') as pp:
        json.dump(parameters_dict, pp, indent=2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='It creates `ni` images with `nl` of lines/segments')
    parser.add_argument('-o', '--output', type=str, default='', help='output folder')
    parser.add_argument('-i', '--images', type=str, default='', help='images folder (where to cut the pieces from)')
    parser.add_argument('-np', '--num_pieces', type=int, default=9, help='number of pieces the images')
    parser.add_argument('-s', '--shape', type=str, default='irregular', help='shape of the pieces', choices=['regular', 'irregular'])
    parser.add_argument('-r', '--rescale', type=int, default=1000, help='rescale the largest of the two axis to this number (default 1000) to avoid large puzzles.')
    args = parser.parse_args()
    main(args)


# ###############################
# num_images = 10
# num_lines = 40
# line_type = 'lines'
# # height = 1000
# # width = 1000
# height = cfg.img_size
# width = cfg.img_size
# patch_size = cfg.piece_size
# num_patches_side = cfg.num_patches_side
# n_patches = num_patches_side * num_patches_side
# #################################

# dataset_path = os.path.join(f'C:\\Users\Marina\OneDrive - unive.it\RL\data')
# cur_folder = os.path.join(dataset_path, f'random_{num_lines}_{line_type}_exact_detection')
# os.makedirs(cur_folder, exist_ok=True)


# print(f'creating images with {num_lines} lines', end='\r')
# for N in range(num_images):
#     ## create images with lines
#     img, all_lines = create_random_image2(line_type, num_lines, height, width)



# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser(description='It creates images with min, ..., max number of lines/segments (from min to max)')
#     parser.add_argument('-min', '--min_lines', type=int, default=1, help='min number of lines')
#     parser.add_argument('-max', '--max_lines', type=int, default=10, help='max number of lines')
#     parser.add_argument('-hh', '--height', type=int, default=1920, help='height of the images')
#     parser.add_argument('-ww', '--width', type=int, default=1920, help='width of the images')
#     parser.add_argument('-i', '--imgs_per_line', type=int, default=10, help='number of images for each number of line')
#     parser.add_argument('-o', '--output', type=str, default='', help='output folder')
#     parser.add_argument('-t', '--type', type=str, default='segments', choices=['segments', 'lines', 'polylines'], help='choose type of features')
#     parser.add_argument('-th', '--thickness', type=int, default=1, help='thickness of the drawings')
#
#     args = parser.parse_args()
#     main(args)