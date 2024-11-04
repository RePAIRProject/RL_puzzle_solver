import numpy as np 
from puzzle_utils.shape_utils import compute_SDF_cost_matrix
from puzzle_utils.lines_ops import compute_cost_matrix_LAP_debug, compute_cost_matrix_LAP, \
        compute_cost_matrix_LAP_v2, compute_cost_matrix_LAP_v3, compute_cost_matrix_LCI_method, \
        extract_from, compute_cost_matrix_LAP_vis
from compatibility.compatibility_Motifs import compute_cost_using_motifs_compatibility
from compatibility.compatibility_MGC import compute_cost_using_color_compatibility
import time


class CfgParameters(dict):
    __getattr__ = dict.__getitem__

def calc_computation_parameters(parameters, cmp_type, cmp_cost, lines_det_method, motif_det_method):

    cmp_pars = CfgParameters()

    cmp_pars['cmp_type'] = cmp_type 
    cmp_pars['cmp_cost'] = cmp_cost 
    cmp_pars['lines_det_method'] = lines_det_method 
    cmp_pars['motif_det_method'] = motif_det_method 
    if cmp_type == 'lines':
        cmp_pars['thr_coef'] = 0.13
        #lm_pars['max_dist'] = 0.70*parameters.xy_step ## changed *0.7
        if (parameters.xy_step)> 6:
            cmp_pars['max_dist'] = 6   ## changed *0.7*parameters.xy_step
        else:
            cmp_pars['max_dist'] = 0.70*(parameters.xy_step)

        cmp_pars['badmatch_penalty'] = max(5, cmp_pars['max_dist'] * 5 / 3) # parameters.piece_size / 3 #?
        cmp_pars['mismatch_penalty'] = max(4, cmp_pars['max_dist'] * 4 / 3) # parameters.piece_size / 4 #?
        cmp_pars['rmax'] = .5 * cmp_pars['max_dist'] * 7 / 6
        cmp_pars['cmp_cost'] = cmp_cost
        cmp_pars['k'] = 3
    elif cmp_type == 'shape':
        cmp_pars['dilation_size'] = 35
    #elif cmp_type == 'motifs':
        #elif cmp_type == 'motifs': #nothing needed it seems

    return cmp_pars

def compute_cost_wrapper(idx1, idx2, pieces, regions_mask, ppars, detector=None, seg_len=0, verbosity=1):
    """
    Wrapper for the cost computation, so that it can be called in one-line, 
    making it easier to parallelize using joblib's Parallel (in comp_irregular.py) 

    # shape branch
    added a "compatibility_type" parameter which allows to control which compatibility to use:
    shape, color, line, pattern.. 
    """

    p = ppars['p']
    m = ppars['m']
    z_id = ppars['z_id']
    rot = ppars['rot']
    n = len(pieces)
    compatibility_type = ppars['cmp_type']
    compatibility_cost = ppars['cmp_cost']
    lines_det_method = ppars['lines_det_method']
    motif_det_method = ppars['motif_det_method']
    
    if verbosity > 1:
        print(f"Computing cost for pieces {idx1:>2} and {idx2:>2}")

    if idx1 == idx2:
        #print('idx == ')
        R_cost = np.zeros((m.shape[1], m.shape[1], len(rot))) - 1
    else:
        poly1 = pieces[idx1]['polygon']
        poly2 = pieces[idx2]['polygon']
        mask_ij = regions_mask[:, :, :, idx2, idx1]
        candidate_values = np.sum(mask_ij > 0)
        if compatibility_type == 'lines':
            alfa1, r1, s11, s12, color1, cat1 = extract_from(pieces[idx1]['extracted_lines'])
            alfa2, r2, s21, s22, color2, cat2 = extract_from(pieces[idx2]['extracted_lines'])
            if len(alfa1) == 0 and len(alfa2) == 0:
                #print('no lines')
                R_cost = np.zeros((m.shape[1], m.shape[1], len(rot))) + ppars.max_dist * 2
            elif (len(alfa1) > 0 and len(alfa2) == 0) or (len(alfa1) == 0 and len(alfa2) > 0):
                #print('only one side with lines')
                R_cost = np.zeros((m.shape[1], m.shape[1], len(rot))) + ppars.mismatch_penalty
            else:
                #print('values!')
                t1 = time.time()
                if compatibility_cost == 'DEBUG':
                    print(f"Computing compatibility between Piece {idx1} and Piece {idx2}")
                    if idx2 - idx1 == 1:
                        plt.suptitle(f"COST between Piece {idx1} and Piece {idx2}", fontsize=32)
                        R_cost = compute_cost_matrix_LAP_debug(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, color1, color2, cat1, cat2,
                                                    mask_ij, ppars, verbosity=verbosity, show=True)
                    else:
                        R_cost = compute_cost_matrix_LAP_debug(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, color1, color2, cat1, cat2, 
                                                    mask_ij, ppars, verbosity=verbosity, show=False)
                elif compatibility_cost == 'LAP':
                    R_cost = compute_cost_matrix_LAP(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, color1, color2, cat1, cat2, 
                                                    mask_ij, ppars, verbosity=verbosity)
                elif compatibility_cost == 'LCI':
                    R_cost = compute_cost_matrix_LCI_method(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, color1, color2, cat1, cat2, 
                                                            mask_ij, ppars, verbosity=verbosity)
                elif compatibility_cost == 'LAP2':
                    R_cost = compute_cost_matrix_LAP_v2(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, color1, color2, cat1, cat2, 
                                                            mask_ij, ppars, verbosity=verbosity)
                elif compatibility_cost == 'LAP3':
                    R_cost = compute_cost_matrix_LAP_v3(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, color1, color2, cat1, cat2, 
                                                            mask_ij, ppars, verbosity=verbosity)
                elif compatibility_cost == 'LAPvis':
                    lines_pi = alfa1, r1, s11, s12, color1, cat1
                    lines_pj = alfa2, r2, s21, s22, color2, cat2 
                    piece_i = pieces[idx1]
                    piece_j = pieces[idx2]
                    R_cost = compute_cost_matrix_LAP_vis(z_id, rot, lines_pi, lines_pj, piece_i, piece_j, mask_ij, ppars, verbosity=1)
                else:
                    print('weird: using {compatibility_cost} method, not known! We use `new` as we dont know what else to do! change --cmp_cost')
                    R_cost = compute_cost_matrix_LCI_method(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, color1, color2, cat1, cat2, 
                                                            mask_ij, ppars)
                if verbosity > 1:
                    print(f"computed cost matrix for piece {idx1} ({len(alfa1)} lines) vs piece {idx2} ({len(alfa2)} lines): took {(time.time()-t1):.02f} seconds ({candidate_values:.1f} candidate values) ")
                #print(R_cost)
        elif compatibility_type == 'shape':
            #breakpoint()
            ids_to_score = np.where(mask_ij > 0)
            R_cost = compute_SDF_cost_matrix(pieces[idx1], pieces[idx2], ids_to_score, ppars, verbosity=verbosity)
            #breakpoint()
        elif compatibility_type == 'motifs':
            assert ( (motif_det_method == "yolo-obb") | (motif_det_method == "yolo-bbox")), f"Unkown detection method for motifs!\nWe know `yolo-obb` and `yolo-bbox`, given `{motif_det_method}`\nRe-run specifying `--det_method`"
            R_cost = compute_cost_using_motifs_compatibility(idx1, idx2, pieces, mask_ij, ppars, yolo_obj_detector=detector, det_type=motif_det_method, verbosity=verbosity)
        elif compatibility_type == 'color':
            R_cost = compute_cost_using_color_compatibility(idx1, idx2, pieces, mask_ij, ppars, seg_len=seg_len, verbosity=1)
        else: # other compatibilities!
            print("\n" * 20)
            print("=" * 50)
            print("WARNING:")
            print(f"Received: {compatibility_type} as compatibility_type")
            print("NOT IMPLEMENTED YET, RETURNING JUST EMPTY MATRIX")
            print("=" * 50)
            print("\n" * 20)

            R_cost = np.zeros((m.shape[1], m.shape[1], len(rot)))
        
    return R_cost

def normalize_cost_matrix(cost_matrix, line_matching_parameters, cmp_cost='LAP'):

    if cmp_cost == 'LCI':
        print("WARNING: normalized over each piece!")
        #All_norm_cost = cost_matrix/np.max(cost_matrix)  # normalize to max value TODO !!!
    elif cmp_cost == 'LAP3':
        min_vals = []
        for j in range(cost_matrix.shape[3]):
            for i in range(cost_matrix.shape[4]):
                min_val = np.min(cost_matrix[:, :, :, j, i])
                min_vals.append(min_val)
        kmin_cut_val = np.max(min_vals) + 1
        All_norm_cost = np.maximum(1 - cost_matrix/ kmin_cut_val, 0)
    elif cmp_cost == 'LAP2':
        clipping_val = line_matching_parameters.max_dist + (line_matching_parameters.badmatch_penalty - line_matching_parameters.max_dist) / 3
        cost_matrix = np.clip(cost_matrix, 0, clipping_val)
        All_norm_cost = 1 - cost_matrix / clipping_val
    else:  # args.cmp_cost == 'LAP':
        #All_norm_cost = np.maximum(1 - cost_matrix / line_matching_parameters.rmax, 0)
        All_norm_cost = cost_matrix # / np.max(cost_matrix) #
    # if zeros_as_negative == True:
    return All_norm_cost

def normalize_CM(R, norm_method=''):
    """
    It normalizes a compatibility matrix with a known structure (-1, 0, positive values) 
    """
    if norm_method == '': # standard
        # since 0 means "far away" we leave lower values 
        # min_val = np.min(R[R > 0])
        # R[R > 0] -= min_val # moved min val to zero
        prm = (R > 0).astype(int)
        max_val = np.max(R[R > 0])
        scaling_factor = np.ones_like(R) * prm * max_val
        import matplotlib.pyplot as plt 
        plt.subplot(121)
        plt.imshow(R[:,:,0,1,2])
        plt.subplot(122)
        plt.imshow(scaling_factor[:,:,0,1,2])
        plt.show()
        breakpoint()
        R /= scaling_factor
    
    return R
