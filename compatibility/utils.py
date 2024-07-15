import numpy as np 
from puzzle_utils.shape_utils import compute_SDF_cost_matrix
from puzzle_utils.lines_ops import compute_cost_matrix_LAP_debug, compute_cost_matrix_LAP, \
        compute_cost_matrix_LAP_v2, compute_cost_matrix_LAP_v3, compute_cost_matrix_LCI_method, \
        extract_from
import time

def compute_cost_wrapper(idx1, idx2, pieces, regions_mask, cmp_parameters, ppars, compatibility_type='lines', verbosity=1):
    """
    Wrapper for the cost computation, so that it can be called in one-line, 
    making it easier to parallelize using joblib's Parallel (in comp_irregular.py) 

    # shape branch
    added a "compatibility_type" parameter which allows to control which compatibility to use:
    shape, color, line, pattern.. 
    """

    (p, z_id, m, rot, line_matching_pars) = cmp_parameters
    n = len(pieces)
    
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
                R_cost = np.zeros((m.shape[1], m.shape[1], len(rot))) + line_matching_pars.max_dist * 2
            elif (len(alfa1) > 0 and len(alfa2) == 0) or (len(alfa1) == 0 and len(alfa2) > 0):
                #print('only one side with lines')
                R_cost = np.zeros((m.shape[1], m.shape[1], len(rot))) + line_matching_pars.mismatch_penalty
            else:
                #print('values!')
                
                t1 = time.time()
                if line_matching_pars.cmp_cost == 'DEBUG':
                    print(f"Computing compatibility between Piece {idx1} and Piece {idx2}")
                    if idx2 - idx1 == 1:
                        plt.suptitle(f"COST between Piece {idx1} and Piece {idx2}", fontsize=32)
                        R_cost = compute_cost_matrix_LAP_debug(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, color1, color2, cat1, cat2, line_matching_pars,
                                                    mask_ij, ppars, verbosity=verbosity, show=True)
                    else:
                        R_cost = compute_cost_matrix_LAP_debug(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, color1, color2, cat1, cat2, line_matching_pars,
                                                    mask_ij, ppars, verbosity=verbosity, show=False)
                elif line_matching_pars.cmp_cost == 'LAP':
                    R_cost = compute_cost_matrix_LAP(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, color1, color2, cat1, cat2, line_matching_pars,
                                                    mask_ij, ppars, verbosity=verbosity)
                elif line_matching_pars.cmp_cost == 'LCI':
                    R_cost = compute_cost_matrix_LCI_method(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, color1, color2, cat1, cat2, line_matching_pars,
                                                            mask_ij, ppars, verbosity=verbosity)
                elif line_matching_pars.cmp_cost == 'LAP2':
                    R_cost = compute_cost_matrix_LAP_v2(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, color1, color2, cat1, cat2, line_matching_pars,
                                                            mask_ij, ppars, verbosity=verbosity)
                elif line_matching_pars.cmp_cost == 'LAP3':
                    R_cost = compute_cost_matrix_LAP_v3(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, color1, color2, cat1, cat2, line_matching_pars,
                                                            mask_ij, ppars, verbosity=verbosity)
                else:
                    print('weird: using {line_matching_pars.cmp_cost} method, not known! We use `new` as we dont know what else to do! change --cmp_cost')
                    R_cost = compute_cost_matrix_LCI_method(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, color1, color2, cat1, cat2, line_matching_pars,
                                                            mask_ij, ppars)
                if verbosity > 1:
                    print(f"computed cost matrix for piece {idx1} ({len(alfa1)} lines) vs piece {idx2} ({len(alfa2)} lines): took {(time.time()-t1):.02f} seconds ({candidate_values:.1f} candidate values) ")
                #print(R_cost)
        elif compatibility_type == 'shape':
            #breakpoint()
            ids_to_score = np.where(mask_ij > 0)
            R_cost = compute_SDF_cost_matrix(pieces[idx1], pieces[idx2], ids_to_score, cmp_parameters, ppars)
            #breakpoint()

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
