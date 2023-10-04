import os 
import cv2 
import random 
import argparse
from geometry import create_random_image
import pdb 

def main(args):

    if args.output == '':
        dataset_path = os.path.join(os.getcwd(), 'data', args.type)
    else:
        dataset_path = args.output

    for num_lines in range(args.min_lines, args.max_lines):
        cur_folder = os.path.join(dataset_path, f'images_with_{num_lines}_{args.type}')
        os.makedirs(cur_folder, exist_ok=True)
        print(f'creating images with {num_lines} {args.type}..', end='\r')
        for k in range(args.imgs_per_line):
            img = create_random_image(args.type, num_lines, args.width, args.height)
            cv2.imwrite(os.path.join(cur_folder, f'image_{k}.jpg'), img)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='It creates images with min, ..., max number of lines/segments (from min to max)')

    parser.add_argument('-min', '--min_lines', type=int, default=1, help='min number of lines')
    parser.add_argument('-max', '--max_lines', type=int, default=10, help='max number of lines')
    parser.add_argument('-hh', '--height', type=int, default=1920, help='height of the images')
    parser.add_argument('-ww', '--width', type=int, default=1920, help='width of the images')
    parser.add_argument('-i', '--imgs_per_line', type=int, default=10, help='number of images for each number of line')
    parser.add_argument('-o', '--output', type=str, default='', help='output folder')
    parser.add_argument('-t', '--type', type=str, default='segments', choices=['segments', 'lines', 'polylines'], help='choose type of features')
    parser.add_argument('-th', '--thickness', type=int, default=1, help='thickness of the drawings')

    args = parser.parse_args()
    main(args)