
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.io
from scipy.ndimage import rotate
from PIL import Image
import os
import configs.folder_names as fnames
import argparse
from compatibility.line_matching_NEW_segments import read_info
import configs.solver_cfg as cfg
from puzzle_utils.pieces_utils import calc_parameters
from puzzle_utils.shape_utils import prepare_pieces_v2, create_grid, include_shape_info, place_on_canvas

def initialization(R, anc):
    # Initialize reconstruction plan
    no_grid_points = R.shape[0]
    no_patches = R.shape[3]
    no_rotations = R.shape[2]

    Y = round(0.5 * (no_grid_points - 1) * (no_patches + 1) + 1)
    X = Y
    Z = no_rotations

    # initialize assignment matrix
    p = np.ones((Y, X, Z, no_patches)) / (Y * X)  # uniform
    init_pos = np.zeros((no_patches, 3)).astype(int)

    # place anchored patch (center)
    y0 = round(Y / 2)
    x0 = round(X / 2)
    z0 = 0  # rotation for anchored patch
    p[:, :, :, anc] = 0
    p[y0, x0, :, :] = 0
    p[y0, x0, z0, anc] = 1
    init_pos[anc, :] = ([y0, x0, z0])

    print("P:", p.shape)
    return p, init_pos, x0, y0, z0


def RePairPuzz(R, p, na, verbosity=1):
    faze = 0
    new_anc = []
    na_new = na
    f = 0
    iter = 0
    eps = np.inf

    all_pay = []
    all_sol = []
    all_anc = []
    Y, X, Z, noPatches = p.shape

    print("started solving..")
    while eps != 0 and iter < cfg.Tmax:
        if na_new > na:
            na = na_new
            faze += 1
            p = np.ones((Y, X, Z, noPatches)) / (Y * X)

            for jj in range(noPatches):
                if new_anc[jj, 0] != 0:
                    y = new_anc[jj, 0]
                    x = new_anc[jj, 1]
                    z = new_anc[jj, 2]
                    p[:, :, :, jj] = 0
                    p[y, x, :, :] = 0
                    p[y, x, z, jj] = 1
        if faze == 0:
            T = cfg.Tfirst
        else:
            T = cfg.Tnext
        p, payoff, eps, iter = solver_rot_puzzle(R, p, T, iter, 0, verbosity=verbosity)

        I = np.zeros((noPatches, 1))
        m = np.zeros((noPatches, 1))

        for j in range(noPatches):
            pj_final = p[:, :, :, j]
            m[j, 0], I[j, 0] = np.max(pj_final), np.argmax(pj_final)

        I = I.astype(int)
        i1, i2, i3 = np.unravel_index(I, p[:, :, :, 1].shape)

        fin_sol = np.concatenate((i1, i2, i3), axis=1)
        if verbosity > 0:
            print("#" * 70)
            print("ITERATION", iter)
            print("#" * 70)
            print(np.concatenate((fin_sol, np.round(m * 100)), axis=1))
        else:
            if iter % 1000 == 0:
                print("iteration", iter)
        a = (m > cfg.anc_fix_tresh).astype(int)
        new_anc = np.array(fin_sol*a)
        na_new = np.sum(a)
        if verbosity > 0:
            print(new_anc)
        f += 1
        all_pay.append(payoff)
        all_sol.append(fin_sol)
        all_anc.append(new_anc)

    # all_sol.append(fin_sol)
    p_final = p
    return all_pay, all_sol, all_anc, p_final, eps, iter, na_new


def solver_rot_puzzle(R, p, T, iter, visual, verbosity=1):
    no_rotations = R.shape[2]
    no_patches = R.shape[3]
    payoff = np.zeros(T+1)
    z_st = 360 / no_rotations
    z_rot = np.arange(0, 360 - z_st + 1, z_st)
    t = 0
    eps = np.inf
    while t < T and eps > 0:
        t += 1
        iter += 1
        q = np.zeros_like(p)
        for i in range(no_patches):
            ri = R[:, :, :, :, i]
            #  ri = R[:, :, :, i, :]  # FOR ORACLE SQUARE ONLY
            for zi in range(no_rotations):
                rr = rotate(ri, z_rot[zi], reshape=False, mode='constant')
                rr = np.roll(rr, zi, axis=2)
                c1 = np.zeros(p.shape)
                for j in range(no_patches):
                    for zj in range(no_rotations):
                        rj_z = rr[:, :, zj, j]
                        pj_z = p[:, :, zj, j]
                        # cc = cv.filter2D(pj_z, -1, np.rot90(rj_z, 2)) # solves in inverse order !?!
                        cc = cv.filter2D(pj_z, -1, rj_z)
                        c1[:, :, zj, j] = cc

                q1 = np.sum(c1, axis=(2, 3))
                # q2 = (q1 != 0) * (q1 + no_patches * no_rotations * 0.5) ## new_experiment
                q2 = (q1 + no_patches * no_rotations * 1)
                q[:, :, zi, i] = q2

        pq = p * q  # e = 1e-11
        p_new = pq / (np.sum(pq, axis=(0, 1, 2)))
        p_new = np.where(np.isnan(p_new), 0, p_new)
        pay = np.sum(p_new * q)

        payoff[t] = pay
        eps = abs(pay - payoff[t-1])
        if verbosity > 0:
            if verbosity == 1:
                print(f'Iteration {t}: pay = {pay:.05f}, eps = {eps:.05f}', end='\r')
            else:
                print(f'Iteration {t}: pay = {pay:.05f}, eps = {eps:.05f}')
        p = np.round(p_new, 8)
    return p, payoff, eps, iter

def reconstruct_puzzle_v2(solved_positions, Y, X, pieces, ppars, use_RGB=True):

    if use_RGB:
        canvas_image = np.zeros((np.round(Y * ppars.xy_step + ppars.p_hs).astype(int), np.round(X * ppars.xy_step + ppars.p_hs).astype(int), 3))
    else:
        canvas_image = np.zeros((np.round(Y * ppars.xy_step + ppars.p_hs).astype(int), np.round(X * ppars.xy_step + ppars.p_hs).astype(int)))
    for i, piece in enumerate(pieces):
        target_pos = solved_positions[i,:2] * ppars.xy_step
        target_rot = solved_positions[i, 2] * ppars.theta_step
        placed_piece = place_on_canvas(piece, target_pos, canvas_image.shape[0], target_rot)
        if use_RGB:
            if len(placed_piece['img'].shape) > 2:
                canvas_image += placed_piece['img']
            else:
                canvas_image += np.repeat(placed_piece['img'], 3).reshape(canvas_image.shape)
        else:
            canvas_image += placed_piece['img']

    return canvas_image

def reconstruct_puzzle(fin_sol, Y, X, pieces, pieces_files, pieces_folder, ppars):
    step = np.ceil(ppars.xy_step)
    ang = 360 / ppars.theta_grid_points
    z_rot = np.arange(0, 360, ang)

    pos = fin_sol
    fin_im = np.zeros(((Y * step + ppars.p_hs).astype(int), (X * step + ppars.p_hs).astype(int), 3))

    for i in range(len(pieces)):
        image = pieces_files[pieces[i]]  # read image 1
        im_file = os.path.join(pieces_folder, image)

        Im0 = Image.open(im_file).convert('RGBA')
        Im = np.array(Im0) / 255.0
        Im1 = Image.open(im_file).convert('RGBA').split()
        alfa = np.array(Im1[3]) / 255.0
        Im = np.multiply(Im, alfa[:, :, np.newaxis])
        Im = Im[:, :, 0:3]

        cc = ppars.p_hs
        ids = (pos[i, :2] * step + cc).astype(int)
        if pos.shape[1] == 3:
            rot = z_rot[pos[i, 2]]
            Im = rotate(Im, rot, reshape=False, mode='constant')
        fin_im[ids[0]-cc:ids[0]+cc, ids[1]-cc:ids[1]+cc, :] = Im+fin_im[ids[0]-cc:ids[0]+cc, ids[1]-cc:ids[1]+cc, :]
    return fin_im


def select_anchor(folder):
    pieces_files = os.listdir(folder)
    json_files = [piece_file for piece_file in pieces_files if piece_file[-4:] == 'json']
    json_files.sort()
    n = len(json_files)

    num_lines = np.zeros(n)
    for f in range(n):
        im = json_files[f]
        beta, R, s1, s2, b1, b2 = read_info(folder, im)
        num_lines[f] = len(beta)

    mean_num_lines = np.round(np.mean(num_lines))
    good_anchors = np.array(np.where(num_lines > mean_num_lines))
    new_anc = np.random.choice(good_anchors[0, :], 1)

    return new_anc[0]


#  MAIN
def main(args):
    print(os.getcwd())
    dataset_name = args.dataset
    puzzle_name = args.puzzle
    method = args.method
    num_pieces = args.pieces

    pieces, img_parameters = prepare_pieces_v2(fnames, args.dataset, args.puzzle, verbose=True)
    ppars = calc_parameters(img_parameters)

    if num_pieces < 1:
        output_root_folder = fnames.output_dir
    else:
        output_root_folder = f"{fnames.output_dir}_{num_pieces}x{num_pieces}"

    mat = scipy.io.loadmat(os.path.join(output_root_folder, dataset_name, puzzle_name,fnames.cm_output_name, f'CM_lines_{method}.mat'))
    pieces_folder = os.path.join(output_root_folder, dataset_name, puzzle_name, f"{fnames.pieces_folder}")
    only_lines_pieces_folder = os.path.join(output_root_folder, dataset_name, puzzle_name, f"{fnames.lines_output_name}", method, 'lines_only')
    detect_output = os.path.join(output_root_folder, dataset_name, puzzle_name, f"{fnames.lines_output_name}", method)
    R = mat['R_line']

    pieces_files = os.listdir(pieces_folder)
    pieces_files.sort()
    print(pieces_files)
    pieces_excl = []
    # pieces_excl = np.array([3,4,7,8,11,15]);
    all_pieces = np.arange(len(pieces_files))
    pieces = [p for p in all_pieces if p not in all_pieces[pieces_excl]]

    pieces_incl = [p for p in np.arange(0, len(all_pieces)) if p not in pieces_excl]
    R = R[:, :, :, pieces_incl, :]  # re-arrange R-matrix
    R = R[:, :, :, :, pieces_incl]

    if args.few_rotations > 0:
        R = R[:, :, :int(args.few_rotations), :, :]

    if args.anchor < 0:
        anc = select_anchor(detect_output)
    else:
        anc = args.anchor
    print(f"Using anchor the piece with id: {anc}")

    p_initial, init_pos, x0, y0, z0 = initialization(R, anc)  # (R, anc, anc_rot, nh, nw)
    print(p_initial.shape)
    na = 1
    all_pay, all_sol, all_anc, p_final, eps, iter, na = RePairPuzz(R, p_initial, na, verbosity=args.verbosity)

    solution_folder = os.path.join(output_root_folder, dataset_name, puzzle_name, f'{fnames.solution_folder_name}_anchor{anc}_{args.method}')
    os.makedirs(solution_folder, exist_ok=True)

    #  SAVE THE MATRIX BEFORE ANY VISUALIZATION
    filename = os.path.join(solution_folder, 'p_final')
    mdic = {"p_final": p_final, "label": "label", "anchor": anc, "anc_position": [x0, y0, z0]}
    scipy.io.savemat(f'{filename}.mat', mdic)
    np.save(filename, mdic)

#   VISUALIZATION
    f = len(all_sol)
    Y, X, Z, _ = p_final.shape
    fin_sol = all_sol[f-1]
    fin_im1 = reconstruct_puzzle(fin_sol, Y, X, pieces, pieces_files, pieces_folder, ppars)
    # alternative method for reconstruction (with transparency on overlap becaus of b/w image)
    fin_im_v2 = reconstruct_puzzle_v2(fin_sol, Y, X, pieces_dict, ppars, use_RGB=False)
    final_solution_v2 = os.path.join(solution_folder, f'final_using_anchor{anc}_overlap.png')
    plt.imsave(final_solution_v2, fin_im_v2)

    os.makedirs(solution_folder, exist_ok=True)
    final_solution = os.path.join(solution_folder, f'final_using_anchor{anc}.png')
    plt.figure(figsize=(16, 16))
    plt.title("Final solution including all piece")
    plt.imshow((fin_im1 * 255).astype(np.uint8))
    plt.tight_layout()
    plt.savefig(final_solution)
    plt.close()

    f = len(all_anc)
    fin_sol = all_anc[f-1]
    fin_im2 = reconstruct_puzzle(fin_sol, Y, X, pieces, pieces_files, pieces_folder, ppars)

    final_solution_anchor = os.path.join(solution_folder, f'final_only_anchor_using_anchor{anc}.png')
    plt.figure(figsize=(16,16))
    plt.title("Final solution including ONLY solved pieces")
    plt.imshow((fin_im2 * 255).astype(np.uint8))
    plt.tight_layout()
    plt.savefig(final_solution_anchor)
    plt.close()

    alc_path = os.path.join(solution_folder, 'alc_plot.png')
    f = len(all_pay)
    f_pay = []
    for ff in range(f):
        a = all_pay[ff]
        f_pay = np.append(f_pay, a)
    f_pay = np.array(f_pay)
    plt.figure(figsize=(6, 6))
    plt.plot(f_pay, 'r', linewidth=1)
    plt.tight_layout()
    plt.savefig(alc_path)

    if args.save_frames is True:
        # intermediate steps
        frames_folders = os.path.join(solution_folder, 'frames_all')
        os.makedirs(frames_folders, exist_ok=True)

        for ff in range(f):
            frame_path = os.path.join(frames_folders, f"frame_{ff:05d}.png")
            cur_sol = all_sol[ff]
            im_rec = reconstruct_puzzle(cur_sol, Y, X, pieces, pieces_files, pieces_folder)
            im_rec = np.clip(im_rec,0,1)
            plt.imsave(frame_path, im_rec)

        frames_folders = os.path.join(solution_folder, 'frames_anc')
        os.makedirs(frames_folders, exist_ok=True)

        for ff in range(f):
            frame_path = os.path.join(frames_folders, f"frame_{ff:05d}.png")
            cur_sol = all_anc[ff]
            im_rec = reconstruct_puzzle(cur_sol, Y, X, pieces, pieces_files, pieces_folder)
            im_rec = np.clip(im_rec,0,1)
            plt.imsave(frame_path, im_rec)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='........ ')  # add some description
    parser.add_argument('--dataset', type=str, default='synthetic_irregular_16_pieces_by_drawing_lines_ruyuvx', help='dataset folder')
    parser.add_argument('--puzzle', type=str, default='', help='puzzle folder')
    # parser.add_argument('--type', type=str, default='irregular', help='puzzle type (regular or irregular)')
    # parser.add_argument('--penalty', type=int, default=20, help='penalty value used')
    parser.add_argument('--method', type=str, default='exact', help='method used for compatibility')  # exact, deeplsd
    parser.add_argument('--pieces', type=int, default=0, help='number of pieces (per side)')
    parser.add_argument('--anchor', type=int, default=0, help='anchor piece (index)')
    parser.add_argument('--save_frames', default=False, action='store_true', help='use to save all frames of the reconstructions')
    parser.add_argument('--verbosity', type=int, default=1, help='level of logging/printing (0 --> nothing, higher --> more printed stuff)')
    parser.add_argument('--few_rotations', type=int, default=0, help='uses only few rotations to make it faster')

    args = parser.parse_args()

    main(args)



# def visualize_result(all_pay, all_sol, all_anc, init_pos, p_final, pieces, pieces_files, pieces_folder, ppars):
#     Y, X, Z, _ = p_final.shape
#     # init_im = reconstruct_toy9(init_pos, Y, X)
#     # init_im = reconstruct_group28_9(init_pos, Y, X, Z, pieces)
#     init_im = reconstruct_puzzle(init_pos, Y, X, pieces, pieces_files, pieces_folder, ppars)
#
#     faze = len(all_sol)
#     col = 2
#     row = faze
#     t = 1
#
#     plt.figure()
#     plt.subplot(col, row, t)
#     plt.imshow((init_im * 255).astype(np.uint8))
#
#     for f in range(faze - 1):
#         t += 1
#         new_anc = all_anc[f]
#         # faze_im = reconstruct_toy9(new_anc, Y, X)
#         # faze_im = reconstruct_group28_9(new_anc, Y, X, Z, pieces)
#         faze_im = reconstruct_puzzle(new_anc, Y, X, pieces, pieces_files, pieces_folder, ppars)
#         plt.subplot(col, row, t)
#         plt.imshow((faze_im * 255).astype(np.uint8))
#
#     for f in range(faze):
#         t += 1
#         fin_sol = all_sol[f]
#         if fin_sol.size != 0:
#             # faze_im = reconstruct_toy9(fin_sol, Y, X)
#             # faze_im = reconstruct_group28_9(fin_sol, Y, X, Z, pieces)
#             faze_im = reconstruct_puzzle(fin_sol, Y, X, pieces, pieces_files, pieces_folder, ppars)
#             plt.subplot(col, row, t)
#             plt.imshow((faze_im * 255).astype(np.uint8))
#     plt.show()
#
#     plt.figure()
#     plt.imshow((faze_im * 255).astype(np.uint8))
#     plt.show()
#
#     f_pay = []
#     for ff in range(faze):
#         a = all_pay[ff]
#         f_pay = np.append(f_pay, a)
#     f_pay = np.array(f_pay)
#     plt.figure()
#     plt.plot(f_pay, 'r', linewidth=1)
#     plt.show()

# def reconstruct_group28_9(fin_sol, Y, X, n_rot, pieces):
#     step = 38
#     #pieces = [p for p in pieces if p not in pieces[pieces_excl]]
#     ang = 360 / n_rot
#     z_rot = np.arange(0, 360, ang)
#     pos = fin_sol
#     fin_im = np.zeros((Y * step + 1000, X * step + 1000, 3))
#
#     for i in range(pos.shape[0]):
#         im_num = pieces[i]
#         in_file = f'C:/Users/Marina/PycharmProjects/WP3-PuzzleSolving/Compatibility/data/repair/group_28/ready/RPf_00{im_num}.png'
#         Im0 = Image.open(in_file).convert('RGBA')
#         Im = np.array(Im0) / 255.0
#         Im1 = Image.open(in_file).convert('RGBA').split()
#         alfa = np.array(Im1[3]) / 255.0
#         Im = np.multiply(Im, alfa[:, :, np.newaxis])
#         Im = Im[:,:,0:3]
#
#         id = pos[i, :2] * step - step + 500
#         if np.min(pos[i, :2]) > 0:
#             if pos.shape[1] == 3:
#                 rot = z_rot[pos[i, 2]]
#                 Im = rotate(Im, rot, reshape=False, mode='constant')
#
#             fin_im[id[0] - 500:id[0] + 500, id[1] - 500:id[1] + 500, :] = Im + fin_im[id[0] - 500:id[0] + 500,
#                                                                                id[1] - 500:id[1] + 500, :]
#     return fin_im