import os, json, pdb, argparse
import shapely
from shapely.affinity import rotate
import cv2
import numpy as np
import matplotlib.pyplot as plt
from configs import folder_names as fnames
# import helper functions
from puzzle_utils.dataset_gen import generate_random_point, create_random_coloured_image, randomword
from puzzle_utils.lines_ops import line_cart2pol
from puzzle_utils.pieces_utils import cut_into_pieces, save_transformation_info, rescale_image
from puzzle_utils.shape_utils import process_region_map
from puzzle_utils.dxf_maps import extract_image_from_dxf

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
        output_root_path = os.path.join(os.getcwd())
    else:
        output_root_path = args.output

    
    if args.shape == 'pattern':
        folder_path = args.patterns_folder
        if folder_path.find("/") > 0:
            folder_name = folder_path.split("/")[-1]
        else:
            folder_name = folder_path
        descr_string = f"{folder_name}"
    elif args.shape == 'square':
        descr_string = f'{args.shape}_{args.num_pieces**2}'
    else:
        descr_string = f'{args.shape}_{args.num_pieces}'
    dataset_name = f"maps_puzzle_{descr_string}_pieces_{randomword(6)}"
    data_folder = os.path.join(output_root_path, fnames.data_path, dataset_name)
    puzzle_folder = os.path.join(output_root_path, fnames.output_dir, dataset_name)
    parameter_path = os.path.join(puzzle_folder, 'parameters.json')
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(puzzle_folder, exist_ok=True)
    use_rotation = (not args.do_not_rotate)

    # save parameters
    parameters_dict = vars(args) # create dict from namespace
    parameters_dict['output'] = output_root_path
    parameters_dict['data_folder'] = data_folder
    parameters_dict['puzzle_folder'] = puzzle_folder
    parameters_dict['use_rotation'] = use_rotation
    print()
    print("#" * 70)
    print("#   Settings:")
    for parkey in parameters_dict.keys():
        print("# ", parkey, ":", parameters_dict[parkey])
    print("#" * 70)
    print()
    
        
    num_pieces_dict = {}
    img_sizes_dict = {}

    # we create one puzzle for each map
    maps_rel_paths_list = os.listdir(args.input)
    num_images_to_be_created = len(maps_rel_paths_list)
    print(maps_rel_paths_list)

    if args.shape == 'pattern':
        list_of_patterns_names = os.listdir(args.patterns_folder)
        num_patterns = len(list_of_patterns_names)
        if num_patterns == num_images_to_be_created:
            print(f"We have {num_images_to_be_created} maps and {num_patterns} pattern, good match!")
        elif num_patterns == 1:
            list_of_patterns_names = list_of_patterns_names * num_images_to_be_created
            print("Only one pattern, repeating it for all maps")
        else:
            print(f"wrong number of patterns ({num_patterns}): either provide 1 or the same number as the maps ({num_images_to_be_created}).")
            pdb.set_trace()

    for N in range(num_images_to_be_created):

        ## create images with lines
        map_rel_path = maps_rel_paths_list[N]
        img, all_lines, sem_categories = extract_image_from_dxf(os.path.join(args.input, map_rel_path))
        if args.rescale > 0:
            img, all_lines = rescale_image(img, args.rescale, all_lines)
        ## save created image
        cv2.imwrite(os.path.join(data_folder, f'image_{N:05d}.jpg'), img)

        # only for patterns
        if args.shape == 'pattern':
            region_map = cv2.imread(os.path.join(args.patterns_folder, list_of_patterns_names[N]), 0)
            pattern_map, num_pieces = process_region_map(region_map)
            print(f"found {num_pieces} pieces on {list_of_patterns_names[N]}")
        else:
            num_pieces = args.num_pieces
        
        ## make folders to save pieces and detected lines
        print("-" * 50)
        puzzle_name = f'image_{N:05d}'
        print(puzzle_name)
        single_image_folder = os.path.join(puzzle_folder, f'image_{N:05d}')
        scaled_image_folder = os.path.join(single_image_folder, fnames.ref_image)
        os.makedirs(scaled_image_folder, exist_ok=True)
        cv2.imwrite(os.path.join(scaled_image_folder, f"output_image_{N:05d}.png"), img)
        pieces_single_folder = os.path.join(single_image_folder, fnames.pieces_folder)
        os.makedirs(pieces_single_folder, exist_ok=True)
        masks_single_folder = os.path.join(single_image_folder, fnames.masks_folder)
        os.makedirs(masks_single_folder, exist_ok=True)
        poly_single_folder = os.path.join(single_image_folder, fnames.polygons_folder)
        os.makedirs(poly_single_folder, exist_ok=True)
        lines_output_folder = os.path.join(single_image_folder, fnames.lines_output_name, 'exact')
        os.makedirs(lines_output_folder, exist_ok=True)
            
        # this gives us the pieces
        if args.shape == 'pattern':
            pieces, patch_size = cut_into_pieces(img, args.shape, num_pieces, single_image_folder, puzzle_name, patterns_map=pattern_map, rotate_pieces=use_rotation, save_extrapolated_regions=args.extrapolation)
        else:
            pieces, patch_size = cut_into_pieces(img, args.shape, num_pieces, single_image_folder, puzzle_name, rotate_pieces=use_rotation, save_extrapolated_regions=args.extrapolation)
        
        ground_truth_path = os.path.join(single_image_folder, f"{fnames.ground_truth}.json")
        save_transformation_info(pieces, ground_truth_path)
        
        for j, piece in enumerate(pieces):
            ## save patch
            cv2.imwrite(os.path.join(pieces_single_folder, f"piece_{j:04d}.png"), piece['squared_image'])
            cv2.imwrite(os.path.join(masks_single_folder, f"piece_{j:04d}.png"), piece['squared_mask'] * 255)
            np.save(os.path.join(poly_single_folder, f"piece_{j:04d}"), piece['squared_polygon'])

            ## we create the container for the lines
            angles = []
            dists = []
            p1s = []
            p2s = []
            b1s = []
            b2s = []
            cols = []
            cats = []
            
            ## check lines and store them
            for i in range(len(all_lines)):
                line = shapely.LineString([all_lines[i, 0:2], all_lines[i, 2:4]])
                col_line = all_lines[i, 4:7].tolist()
                cat = all_lines[i, 7]
                intersect = shapely.intersection(line, piece['polygon'])  # points of intersection line with the piece
                if not shapely.is_empty(intersect): # we have intersection, so the line is important for this piece
                    intersection_lines = []
                    # Assume 'multiline' is your MultiLineString object
                    if isinstance(intersect, shapely.geometry.MultiLineString):
                        # 'multiline' is a MultiLineString
                        intersection_lines = intersect.geoms
                    elif isinstance(intersect, shapely.geometry.LineString):
                        intersection_lines = [intersect]
                    # else: # skip 
                    if len(intersection_lines) > 0: 
                        for int_line in intersection_lines:
                            if len(list(zip(*int_line.xy))) > 1: # two intersections meaning it crosses
                                # list(zip) re-order xs and ys
                                # # xs, ys = list(zip(*int_line.xy))
                                # .xy returns array of x and array of y values
                                xs, ys = int_line.xy
                                p1 = np.array(np.round([xs[0], ys[0]]).astype(int))  ## CHECK !!!
                                p2 = np.array(np.round([xs[1], ys[1]]).astype(int))  ## CHECK !!!
                                rho, theta = line_cart2pol(p1, p2)
                                angles.append(theta)
                                dists.append(rho)
                                p1s.append(p1.tolist())
                                p2s.append(p2.tolist())
                                cols.append(col_line)
                                cats.append(cat)
                            # if len(list(zip(*int_line.xy))) > 1: # two intersections meaning it crosses
                            #     xs, ys = list(zip(*int_line.xy))
                            #     pdb.set_trace()
                            #     p1 = np.array(np.round(pi1).astype(int))  ## CHECK !!!
                            #     p2 = np.array(np.round(pi2).astype(int))  ## CHECK !!!
                            #     rho, theta = line_cart2pol(p1, p2)
                            #     angles.append(theta)
                            #     dists.append(rho)
                            #     p1s.append(p1.tolist())
                            #     p2s.append(p2.tolist())
                            #     cols.append(col_line)

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
                'colors': cols,
                'categories': cats
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
            if 'rotation' in piece.keys():
                # piece['rotation'] is in degrees!
                rot_origin = [piece['squared_image'].shape[0] // 2, piece['squared_image'].shape[1] // 2]
                rotated_square_angles = []
                rotated_squared_p1s = []
                rotated_squared_p2s = []
                for i in range(len(squared_angles)):
                    p1 = squared_p1s[i]
                    p2 = squared_p2s[i]
                    ang = squared_angles[i]
                    shp_line = shapely.LineString([p1, p2])
                    # shapely uses negative angle (in degrees)
                    shp_rotated_line = rotate(shp_line, -piece['rotation'], origin=rot_origin)
                    xs, ys = shp_rotated_line.xy
                    rotated_squared_p1s.append([xs[0], ys[0]])
                    rotated_squared_p2s.append([xs[1], ys[1]])
                    # this is in rad
                    rotated_square_angles.append(ang + np.deg2rad(piece['rotation']))
                # replace with rotated values!
                squared_angles = rotated_square_angles
                squared_p1s = rotated_squared_p1s
                squared_p2s = rotated_squared_p2s

            # plt.imshow(piece['squared_image'])
            # plt.plot(*piece['squared_polygon'].boundary.xy)
            # plt.show()
            # pdb.set_trace()
            # dists, b1 and b2 are not used
            aligned_lines = {
                'angles': squared_angles,   # polar coordinates
                'dists': [],                # distance polar coordinates (not used)
                'p1s': squared_p1s,         # cartesian coordinates
                'p2s': squared_p2s,         # cartesian coordinates
                'b1s': [],
                'b2s': [],
                'colors': cols,
                'categories': cats
            }
            if args.save_visualization:
                sq_lines_only = os.path.join(lines_output_folder, 'squared_lines')
                os.makedirs(sq_lines_only, exist_ok=True)
                len_lines = len(squared_angles)
                sq_lines_img = np.zeros(shape=piece['squared_image'].shape, dtype=np.uint8)
                sq_lines_only_transparent = np.zeros((sq_lines_img.shape[0], sq_lines_img.shape[1], 4))
                sq_lines_only_transparent[:,:,3] = piece['squared_mask']
                if len_lines > 0:
                    plt.figure()
                    plt.title(f'extracted {len_lines} segments')
                    plt.imshow(piece['squared_image'])
                    for p1, p2 in zip(p1s, p2s):
                        plt.plot((p1[0], p2[0]), (p1[1], p2[1]), color='red', linewidth=1)        
                    plt.savefig(os.path.join(sq_lines_only, f"piece_{j:04d}.jpg"))
                    plt.close()
                else:
                    plt.title('no lines')
                    plt.imshow(piece['squared_image'])    
                    plt.savefig(os.path.join(sq_lines_only, f"piece_{j:04d}.jpg"))
                    plt.close()
            with open(os.path.join(lines_output_folder, f"piece_{j:04d}.json"), 'w') as lj:
                json.dump(aligned_lines, lj, indent=3)
            print(f'done with image_{N:05d}/piece_{j:04d}')
        
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
        print(f"Done with {puzzle_name}: created {j+1} pieces.")
    
    parameters_dict['num_pieces'] = num_pieces_dict
    parameters_dict['img_sizes'] = img_sizes_dict
    with open(parameter_path, 'w') as pp:
        json.dump(parameter_path, pp, indent=2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='It generates puzzle from .dxf maps')
    # folders
    parser.add_argument('-i', '--input', type=str, default='', help='input folder with the maps')
    parser.add_argument('-o', '--output', type=str, default='', help='output folder')
    # cutting pieces 
    parser.add_argument('-s', '--shape', type=str, default='irregular', help='shape of the pieces', choices=['square', 'pattern', 'irregular'])
    parser.add_argument('-pf', '--patterns_folder', type=str, default='', help='(used only if shape == pattern): the folder where the patterns are stored')
    parser.add_argument('-r', '--rescale', type=int, default=300, help='rescale the largest of the two axis to this number (default 1000) to avoid large puzzles.')
    parser.add_argument('-np', '--num_pieces', type=int, default=9, help='number of pieces in which each puzzle image is cut')
    parser.add_argument('-sv', "--save_visualization", help="Use it to create visualization", action="store_true")
    parser.add_argument('-noR', "--do_not_rotate", help="Use it to disable rotation!", action="store_true")
    parser.add_argument('-extr', "--extrapolation", help="Use it to create an extrapolated version of each fragment", action="store_true")
    args = parser.parse_args()
    main(args)
