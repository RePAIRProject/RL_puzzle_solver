import os 
import cv2 
import shapely 
from puzzle_utils.puzzle_gen.generator import PuzzleGenerator

def cut_into_pieces(image, shape, num_pieces, output_path, _index):

    pieces = []
    if shape == 'regular':
        patch_size = image.shape[0] // num_pieces
        x0_all = np.arange(0, image.shape[0], patch_size, dtype=int)
        y0_all = np.arange(0, image.shape[1], patch_size, dtype=int)
        for iy in range(num_pieces):
            for ix in range(num_pieces):
                x0 = x0_all[ix]
                y0 = y0_all[iy]
                x1 = x0 + patch_size - 1
                y1 = y0 + patch_size - 1
                box = shapely.box(x0, y0, x1, y1)  # patche box (xmin, ymin, xmax, ymax)
                ## create patch
                patch = image[y0:y1 + 1, x0:x1 + 1]
                centered_patch, centered_mask, shift2align = center_fragment(patch)
                piece_dict = {
                    'center_img': centered_patch,
                    'orig_img': patch,
                    'shape': box,
                    'mask': centered_mask,
                    'shift2center': shift2align
                }
                pieces.append(piece_dict)
                
    if shape == 'irregular':

        generator = PuzzleGenerator(image, f"image_{_index:05d}")
        generator.run(num_pieces, offset_rate_h=0.2, offset_rate_w=0.2, small_region_area_ratio=0.25, rot_range=0,
            smooth_flag=False, alpha_channel=True, perc_missing_fragments=0, erosion=0, borders=False)
        generator.save_jpg_regions(output_path)
        pieces = generator.get_pieces_from_puzzle()
    
    return pieces

def center_fragment(image):
    #pdb.set_trace()
    sd, mask = get_sd(image)
    cm = get_cm(mask)
    center_pos = [np.round(image.shape[0]/2).astype(int), np.round(image.shape[1]/2).astype(int)]
    shift = np.round(np.array(cm) - center_pos).astype(int)
    centered_image = shift_img(image, -shift[0], -shift[1])    
    centered_mask = shift_img(mask, -shift[0], -shift[1])    
    return centered_image, centered_mask, shift

def read_pieces(fnames, puzzle_name):
    """
    Read the pieces and return   
    """
    pieces = []
    data_folder = os.path.join(fnames.data_path, puzzle_name, fnames.imgs_folder)
    pieces_names = os.listdir(data_folder)
    for piece_name in pieces_names:
        piece_full_path = os.path.join(data_folder, piece_name)
        piece_d = {}
        img = cv2.imread(piece_full_path)
        if background != 0:
            img[img[:,:,0] == background] = 0
        if img.shape[0] != img.shape[1]:
            squared_img = np.zeros((cfg.piece_size, cfg.piece_size, 3))
            mx = cfg.piece_size - img.shape[0]
            bx = 0
            if mx % 2 > 0:
                bx = 1
            mx += bx
            by = 0
            my = cfg.piece_size - img.shape[1]
            if my % 2 > 0:
                by = 1    
            my += by
            squared_img[mx//2:-mx//2, my//2:-my//2] = img[:img.shape[0]-bx, :img.shape[1]-by]
            piece_d['img'] = squared_img
        else:
            piece_d['img'] = img
        piece_d['img'] = img
        piece_d['id'] = piece_name[:9]
        pieces.append(piece_d)
    return pieces