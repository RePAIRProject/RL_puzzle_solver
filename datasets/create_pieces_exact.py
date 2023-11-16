import os, json, pdb, argparse
import shapely
import cv2
import numpy as np
import matplotlib.pyplot as plt
# import helper functions
from puzzle_utils.dataset_gen import generate_random_point, create_random_image
from puzzle_utils.lines_ops import line_cart2pol
from puzzle_utils.pieces_utils import cut_into_pieces

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
        output_root_path = os.path.join(os.getcwd(), 'synthetic_dataset')
    else:
        output_root_path = args.output
    
    data_folder = os.path.join(output_root_path, 'data')
    puzzle_folder = os.path.join(output_root_path, 'puzzle')
    parameter_path = os.path.join(output_root_path, 'parameters.json')
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
    with open(parameter_path, 'w') as pp:
        json.dump(parameter_path, pp, indent=2)
    
    for N in range(args.num_images):
        ## create images with lines
        img, all_lines = create_random_image(args.line_type, args.num_lines, args.height, args.width)

        ## save created image
        cv2.imwrite(os.path.join(data_folder, f'image_{N:05d}.jpg'), img)

        ## make folders to save pieces and detected lines
        single_image_folder = os.path.join(puzzle_folder, f'image_{N:05d}')
        pieces_single_folder = os.path.join(single_image_folder, 'pieces')
        os.makedirs(pieces_single_folder, exist_ok=True)
        masks_single_folder = os.path.join(single_image_folder, 'masks')
        os.makedirs(masks_single_folder, exist_ok=True)
        poly_single_folder = os.path.join(single_image_folder, 'polygons')
        os.makedirs(poly_single_folder, exist_ok=True)
        lines_output_folder = os.path.join(single_image_folder, 'lines_detection', 'exact')
        os.makedirs(lines_output_folder, exist_ok=True)

        # this gives us the pieces
        pieces = cut_into_pieces(img, args.shape, args.num_pieces, single_image_folder, N)
        # pieces is a list of dicts with several keys:
        # - pieces[i]['orig_img'] is the image (4-channels, alpha-transparent) in its original location
        # - pieces[i]['center_img'] is the centered image 
        # - pieces[i]['shape'] is the shape (as a shapely polygon)
        # - pieces[i]['shift2center'] is the translation to align the shape to the center of the square

        for j, piece in enumerate(pieces):
            ## save patch
            cv2.imwrite(os.path.join(pieces_single_folder, f"piece_{j:04d}.png"), piece['centered_image'])
            cv2.imwrite(os.path.join(masks_single_folder, f"piece_{j:04d}.png"), piece['centered_mask'] * 255)
            np.save(os.path.join(poly_single_folder, f"piece_{j:04d}"), piece['centered_polygon'])

            ## we create the container for the lines
            angles = []
            dists = []
            p1s = []
            p2s = []
            b1s = []
            b2s = []

            ## lines on the .json file
            for i in range(args.num_lines):
                line = shapely.LineString([all_lines[i, 0:2], all_lines[i, 2:4]])
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
                                p1 = np.array(np.round(pi1 + piece['shift2center']).astype(int))  ## CHECK !!!
                                p2 = np.array(np.round(pi2 + piece['shift2center']).astype(int))  ## CHECK !!! 
                                # p1 = np.array(np.round(pi1 - shift_piece ).astype(int))  ## CHECK !!!
                                # p2 = np.array(np.round(pi2 - shift_piece ).astype(int))  ## CHECK !!!
                                # p1 = np.array(np.round(pi1).astype(int))  ## CHECK !!!
                                # p2 = np.array(np.round(pi2).astype(int))  ## CHECK !!!
                                rho, theta = line_cart2pol(p1, p2)
                                angles.append(theta)
                                dists.append(rho)
                                p1s.append(p1.tolist())
                                p2s.append(p2.tolist())

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
                lines_img = np.zeros(shape=piece['centered_image'].shape, dtype=np.uint8)
                if len_lines > 0:
                    plt.figure()
                    plt.title(f'extracted {len_lines} segments')
                    plt.imshow(piece['centered_image'])
                    for p1, p2 in zip(p1s, p2s):
                        plt.plot((p1[0], p2[0]), (p1[1], p2[1]), color='red', linewidth=1)        
                    plt.savefig(os.path.join(line_vis, f"piece_{j:04d}.jpg"))
                    plt.close()
                    # save one black&white image of the lines
                    #pdb.set_trace()
                    for p1, p2 in zip(p1s, p2s):
                        lines_img = cv2.line(lines_img, np.round(p1).astype(int), np.round(p2).astype(int), color=(255, 255, 255), thickness=1)        
                    cv2.imwrite(os.path.join(lines_only, f"piece_{j:04d}_l.jpg"), 255-lines_img)
                else:
                    plt.title('no lines')
                    plt.imshow(piece['centered_image'])    
                    plt.savefig(os.path.join(line_vis, f"piece_{j:04d}.jpg"))
                    plt.close()
                    cv2.imwrite(os.path.join(lines_only, f"piece_{j:04d}_l.jpg"), 255-lines_img)
            #####


            detected_lines = {
                'angles': angles,
                'dists': dists,
                'p1s': p1s,
                'p2s': p2s,
                'b1s': b1s,
                'b2s': b2s
            }
            with open(os.path.join(lines_output_folder, f"piece_{j:04d}.json"), 'w') as lj:
                json.dump(detected_lines, lj, indent=3)
            print(f'done with image_{N:05d}/piece_{j:04d}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='It creates `ni` images with `nl` of lines/segments')
    parser.add_argument('-nl', '--num_lines', type=int, default=50, help='min number of lines')
    parser.add_argument('-hh', '--height', type=int, default=1920, help='height of the images')
    parser.add_argument('-ww', '--width', type=int, default=1920, help='width of the images')
    parser.add_argument('-ni', '--num_images', type=int, default=10, help='number of images for each number of line')
    parser.add_argument('-o', '--output', type=str, default='', help='output folder')
    parser.add_argument('-lt', '--line_type', type=str, default='segments', choices=['segments', 'lines', 'polylines'], help='choose type of features')
    parser.add_argument('-th', '--thickness', type=int, default=1, help='thickness of the drawings')
    parser.add_argument('-s', '--shape', type=str, default='irregular', help='shape of the pieces', choices=['regular', 'irregular'])
    parser.add_argument('-np', '--num_pieces', type=int, default=9, help='number of pieces the images')
    parser.add_argument('-sv', "--save_visualization", dest="save_visualization", \
        help="Use it to create visualization", action="store_true")
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