import os, json, pdb, argparse
import shapely
import cv2
import numpy as np
import matplotlib.pyplot as plt
# import helper functions
from puzzle_utils.dataset_gen import generate_random_point, create_random_coloured_image
from puzzle_utils.lines_ops import line_cart2pol
from puzzle_utils.pieces_utils import cut_into_pieces
from puzzle_utils.shape_utils import process_region_map
import random, string

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


def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

def main(args):

    if args.output == '':
        output_root_path = os.path.join(os.getcwd())
    else:
        output_root_path = args.output

    
    if args.shape == 'pattern':
        num_pieces_string = ''
    else:
        num_pieces_string = f'{args.num_pieces}_'
    dataset_name = f"synthetic_{args.shape}_{num_pieces_string}pieces_by_drawing_coloured_lines_{randomword(6)}"
    data_folder = os.path.join(output_root_path, 'data', dataset_name)
    puzzle_folder = os.path.join(output_root_path, 'output', dataset_name)
    parameter_path = os.path.join(puzzle_folder, 'parameters.json')
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(puzzle_folder, exist_ok=True)

    # save parameters
    parameters_dict = vars(args) # create dict from namespace
    parameters_dict['output'] = output_root_path
    parameters_dict['data_folder'] = data_folder
    parameters_dict['puzzle_folder'] = puzzle_folder
    print()
    print("#" * 70)
    print("#   Settings:")
    for parkey in parameters_dict.keys():
        print("# ", parkey, ":", parameters_dict[parkey])
    print("#" * 70)
    print()
    
        
    num_pieces_dict = {}
    img_sizes_dict = {}

    if args.shape == 'pattern':
        list_of_patterns_names = os.listdir(args.patterns_folder)
        num_images_to_be_created = len(list_of_patterns_names)
    else:
        num_images_to_be_created = args.num_images

    for N in range(num_images_to_be_created):
        ## create images with lines
        img, all_lines = create_random_coloured_image(args.line_type, args.num_lines, args.height, args.width, num_colors=args.num_colors)

        ## save created image
        cv2.imwrite(os.path.join(data_folder, f'image_{N:05d}.jpg'), img)

        # only for patterns
        if args.shape == 'pattern':
            region_map = cv2.imread(os.path.join(args.patterns_folder, list_of_patterns_names[N]), 0)
            pattern_map, num_pieces = process_region_map(region_map)
        else:
            num_pieces = args.num_pieces
        ## make folders to save pieces and detected lines
        print("-" * 50)
        puzzle_name = f'image_{N:05d}'
        print(puzzle_name)
        single_image_folder = os.path.join(puzzle_folder, f'image_{N:05d}')
        scaled_image_folder = os.path.join(single_image_folder, 'image_scaled')
        os.makedirs(scaled_image_folder, exist_ok=True)
        cv2.imwrite(os.path.join(scaled_image_folder, f"output_image_{N:05d}.png"), img)
        pieces_single_folder = os.path.join(single_image_folder, 'pieces')
        os.makedirs(pieces_single_folder, exist_ok=True)
        masks_single_folder = os.path.join(single_image_folder, 'masks')
        os.makedirs(masks_single_folder, exist_ok=True)
        poly_single_folder = os.path.join(single_image_folder, 'polygons')
        os.makedirs(poly_single_folder, exist_ok=True)
        lines_output_folder = os.path.join(single_image_folder, 'lines_detection', 'exact')
        os.makedirs(lines_output_folder, exist_ok=True)

        # this gives us the pieces
        if args.shape == 'pattern':
            pieces, patch_size = cut_into_pieces(img, args.shape, num_pieces, single_image_folder, puzzle_name, patterns_map=pattern_map, save_extrapolated_regions=args.extrapolation)
        else:
            pieces, patch_size = cut_into_pieces(img, args.shape, num_pieces, single_image_folder, puzzle_name, save_extrapolated_regions=args.extrapolation)
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

            num_pieces_dict[puzzle_name] = j+1
            img_sizes_dict[puzzle_name] = img.shape

            # parameters of the single image!
            img_parameters = {}
            img_parameters['piece_size'] = int(patch_size)
            img_parameters['num_pieces'] = j+1
            img_parameters['size'] = img.shape

            single_image_parameters_path = os.path.join(single_image_folder, f'parameters_{puzzle_name}.json')
            with open(single_image_parameters_path, 'w') as pp:
                json.dump(img_parameters, pp, indent=2)

            ## we create the container for the lines
            angles = []
            dists = []
            p1s = []
            p2s = []
            b1s = []
            b2s = []
            cols = []

            ## lines on the .json file
            for i in range(args.num_lines):
                line = shapely.LineString([all_lines[i, 0:2], all_lines[i, 2:4]])
                col_line = all_lines[i, 4:7].tolist()
                intersect = shapely.intersection(line, piece['polygon'])  # points of intersection line with the piece
                if not shapely.is_empty(intersect): # we have intersection, so the line is important for this piece
                    #print(intersect)
                    intersection_lines = []
                    # Assume 'multiline' is your MultiLineString object
                    if isinstance(intersect, shapely.geometry.MultiLineString):
                        # 'multiline' is a MultiLineString
                        intersection_lines = intersect.geoms
                    elif isinstance(intersect, shapely.geometry.LineString):
                        intersection_lines = [intersect]
                    # else:
                        # skip 
                    if len(intersection_lines) > 0: 
                        for int_line in intersection_lines:
                            #print(int_line)
                            if len(list(zip(*int_line.xy))) > 1: # two intersections meaning it crosses
                                pi1, pi2 = list(zip(*int_line.xy))
                                p1 = np.array(np.round(pi1).astype(int))  ## CHECK !!!
                                p2 = np.array(np.round(pi2).astype(int))  ## CHECK !!!
                                # p1 = np.array(np.round(pi1 + piece['shift2center']).astype(int))  ## CHECK !!!
                                # p2 = np.array(np.round(pi2 + piece['shift2center']).astype(int))  ## CHECK !!! 
                                # p1 = np.array(np.round(pi1 - shift_piece ).astype(int))  ## CHECK !!!
                                # p2 = np.array(np.round(pi2 - shift_piece ).astype(int))  ## CHECK !!!
                                # p1 = np.array(np.round(pi1).astype(int))  ## CHECK !!!
                                # p2 = np.array(np.round(pi2).astype(int))  ## CHECK !!!
                                rho, theta = line_cart2pol(p1, p2)
                                angles.append(theta)
                                dists.append(rho)
                                p1s.append(p1.tolist())
                                p2s.append(p2.tolist())
                                cols.append(col_line)

                            ## OLD VERSION WITH DIFF SHAPE OF FRAGs IMAGE
                            # #print(int_line)
                            # if len(list(zip(*int_line.xy))) > 1: # two intersections meaning it crosses
                            #     pi1, pi2 = list(zip(*int_line.xy))
                            #     #pdb.set_trace()
                            #     shift_piece = np.asarray([piece['corner_x'], piece['corner_y']])
                            #     p1 = np.array(np.round(pi1 - shift_piece + piece['shift2center_frag']).astype(int))  ## CHECK !!!
                            #     p2 = np.array(np.round(pi2 - shift_piece + piece['shift2center_frag']).astype(int))  ## CHECK !!! 
                            #     # p1 = np.array(np.round(pi1 - shift_piece ).astype(int))  ## CHECK !!!
                            #     # p2 = np.array(np.round(pi2 - shift_piece ).astype(int))  ## CHECK !!!
                            #     # p1 = np.array(np.round(pi1).astype(int))  ## CHECK !!!
                            #     # p2 = np.array(np.round(pi2).astype(int))  ## CHECK !!!
                            #     rho, theta = line_cart2pol(p1, p2)
                            #     angles.append(theta)
                            #     dists.append(rho)
                            #     p1s.append(p1.tolist())
                            #     p2s.append(p2.tolist())

            #####
            if args.save_visualization:
                line_vis = os.path.join(lines_output_folder, 'visualization')
                lines_only = os.path.join(lines_output_folder, 'lines_only')
                os.makedirs(line_vis, exist_ok=True)
                os.makedirs(lines_only, exist_ok=True)
                len_lines = len(angles)
                lines_img = np.zeros(shape=piece['image'].shape, dtype=np.uint8)
                lines_only_transparent = np.zeros((lines_img.shape[0], lines_img.shape[1], 4))
                lines_only_transparent[:,:,3] = piece['mask']
                if len_lines > 0:
                    plt.figure()
                    plt.title(f'extracted {len_lines} segments')
                    plt.imshow(piece['image'])
                    for p1, p2 in zip(p1s, p2s):
                        plt.plot((p1[0], p2[0]), (p1[1], p2[1]), color='red', linewidth=1)        
                    plt.savefig(os.path.join(line_vis, f"piece_{j:04d}.jpg"))
                    plt.close()
                    # save one black&white image of the lines
                    #pdb.set_trace()
                    for p1, p2 in zip(p1s, p2s):
                        lines_img = cv2.line(lines_img, np.round(p1).astype(int), np.round(p2).astype(int), color=col_line, thickness=1)        
                    #cv2.imwrite(os.path.join(lines_only, f"piece_{j:04d}_l.jpg"), 255-lines_img)
                    binary_01_mask = (255 - lines_img) / 255
                    lines_only_transparent[:,:,:3] = binary_01_mask
                    plt.imsave(os.path.join(lines_only, f"piece_{j:04d}_t.png"), lines_only_transparent)
                else:
                    plt.title('no lines')
                    plt.imshow(piece['image'])    
                    plt.savefig(os.path.join(line_vis, f"piece_{j:04d}.jpg"))
                    plt.close()
                    #cv2.imwrite(os.path.join(lines_only, f"piece_{j:04d}_l.jpg"), 255-lines_img)
                    plt.imsave(os.path.join(lines_only, f"piece_{j:04d}_t.png"), lines_only_transparent)
            #####
            # THESE ARE THE COORDINATES OF THE ORIGINAL PIECE
            detected_lines = {
                'angles': angles,
                'dists': dists,
                'p1s': p1s,
                'p2s': p2s,
                'b1s': b1s,
                'b2s': b2s,
                'colors': cols
            }
            orig_coords_folder = os.path.join(lines_output_folder, 'original_coords')
            os.makedirs(orig_coords_folder, exist_ok=True)
            with open(os.path.join(orig_coords_folder, f"piece_{j:04d}.json"), 'w') as lj:
                json.dump(detected_lines, lj, indent=3)

            # NOW WE RE-ALIGN THE EXTRACTED LINES TO THE 'SQUARED' img, mask, polygon..
            squared_angles = []
            squared_p1s = []
            squared_p2s = []
            for ang, p1, p2 in zip(angles, p1s, p2s):
                squared_angles.append(ang) # no rotation, but this will change as soon as we introduce rotation
                squared_p1s.append((p1 + piece['shift2center'][::-1] + piece['shift2square'][::-1]).tolist())
                squared_p2s.append((p2 + piece['shift2center'][::-1] + piece['shift2square'][::-1]).tolist())

            # dists, b1 and b2 are not used
            aligned_lines = {
                'angles': squared_angles,
                'dists': [],
                'p1s': squared_p1s,
                'p2s': squared_p2s,
                'b1s': [],
                'b2s': [],
                'colors': cols
            }
            with open(os.path.join(lines_output_folder, f"piece_{j:04d}.json"), 'w') as lj:
                json.dump(aligned_lines, lj, indent=3)
            print(f'done with image_{N:05d}/piece_{j:04d}')
        
        print(f"Done with {puzzle_name}: created {j+1} pieces.")
    
    parameters_dict['num_pieces'] = num_pieces_dict
    parameters_dict['img_sizes'] = img_sizes_dict
    with open(parameter_path, 'w') as pp:
        json.dump(parameter_path, pp, indent=2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='It generates synthetic puzzle by first drawing (colored) segments/lines on an image, \
            then cutting it into pieces and saving pieces and the segments. \
            Check the parameters for details about size, line_type, colors, number of pieces and so on.')
    # line generation
    parser.add_argument('-lt', '--line_type', type=str, default='mix', choices=['segments', 'lines', 'polylines', 'mix'], help='choose type of features')
    parser.add_argument('-nl', '--num_lines', type=int, default=50, help='number of lines drawn in the image')
    parser.add_argument('-ncol', '--num_colors', type=int, default=5, choices=[1, 3, 5], help='number of different colors')
    parser.add_argument('-hh', '--height', type=int, default=300, help='height of the images')
    parser.add_argument('-ww', '--width', type=int, default=300, help='width of the images')
    parser.add_argument('-th', '--thickness', type=int, default=1, help='thickness of the drawings')
    # image generation
    parser.add_argument('-ni', '--num_images', type=int, default=10, help='number of different version of images generated for each number of line')
    parser.add_argument('-o', '--output', type=str, default='', help='output folder')
    # cutting pieces 
    parser.add_argument('-s', '--shape', type=str, default='irregular', help='shape of the pieces', choices=['regular', 'pattern', 'irregular'])
    parser.add_argument('-pf', '--patterns_folder', type=str, default='', help='(used only if shape == pattern): the folder where the patterns are stored')
    parser.add_argument('-np', '--num_pieces', type=int, default=9, help='number of pieces in which each puzzle image is cut')
    parser.add_argument('-sv', "--save_visualization", help="Use it to create visualization", action="store_true")
    parser.add_argument('-extr', "--extrapolation", help="Use it to create visualization", action="store_true")
    args = parser.parse_args()
    main(args)
