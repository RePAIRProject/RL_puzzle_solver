import os 
import cv2 

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