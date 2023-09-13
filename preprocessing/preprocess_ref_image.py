import cv2 
import pdb 
from configs import folder_names as fnames 
from configs import wikiart_cfg as cfg
import os

def main(args):

    image = cv2.imread(args.puzzle)
    image = cv2.resize(image, (cfg.img_size, cfg.img_size))
    num_patches_side = cfg.num_patches_side
    patch_size = image.shape[0] // num_patches_side
    pieces_folder = os.path.join(fnames.data_path, f'wikiart_kuroda_{num_patches_side}x{num_patches_side}', fnames.imgs_folder)
    os.makedirs(pieces_folder, exist_ok=True)
    k = 0
    for i in range(num_patches_side):
        for j in range(num_patches_side):
            patch = image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size,:]
            cv2.imwrite(os.path.join(pieces_folder, f"piece_{k}.png"), patch)
            k += 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create patches from image')
    parser.add_argument('-p', '--puzzle', type=str, default='data/wikiart_kuroda/aki-kuroda_night-2011.jpg', help='image to work on')
    args = parser.parse_args()
    main(args)