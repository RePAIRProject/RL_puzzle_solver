import numpy as np 
from puzzle_utils.shape_utils import compute_SDF_CM_matrix
from puzzle_utils.lines_ops import compute_line_based_CM_LAP, compute_line_based_CM_LCI, \
        extract_from, compute_cost_matrix_LAP_vis, compute_cost_matrix_LAP_debug
from compatibility.compatibility_Motifs import compute_CM_using_motifs
from compatibility.compatibility_MGC import compute_cost_using_color_compatibility
from compatibility.compatibility_Oracle import compute_oracle_compatibility
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
            cmp_pars['max_dist'] = 1.7*(parameters.xy_step)  #REPAIR altrimenti 0.7*parameters.xy_step)

        cmp_pars['badmatch_penalty'] = max(5, cmp_pars['max_dist'] * 5 / 3) # parameters.piece_size / 3 #?
        cmp_pars['mismatch_penalty'] = 1  ## REPAIR
        #cmp_pars['mismatch_penalty'] = max(4, cmp_pars['max_dist'] * 4 / 3) # parameters.piece_size / 4 #?
        cmp_pars['rmax'] = .5 * cmp_pars['max_dist'] * 7 / 6  ##UNUSED
        cmp_pars['cmp_cost'] = cmp_cost
        cmp_pars['k'] = 3
    elif cmp_type == 'shape':
        cmp_pars['dilation_size'] = 35      ##UNUSED

    #elif cmp_type == 'motifs':
        #elif cmp_type == 'motifs': #nothing needed it seems

    return cmp_pars

def compute_cost_wrapper(idx1, idx2, pieces, regions_mask, ppars, puzzle_root_folder, detector=None, seg_len=0, verbosity=1):
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
    
    #if verbosity > 1:
        #print(f"Computing cost for pieces {idx1:>2} and {idx2:>2}")

    compatibility_matrix = np.zeros((m.shape[1], m.shape[1], len(rot)))
    if idx1 != idx2:
        poly1 = pieces[idx1]['polygon']
        poly2 = pieces[idx2]['polygon']
        mask_ij = regions_mask[:, :, :, idx2, idx1]
        candidate_values = np.sum(mask_ij > 0)
        if compatibility_type == 'lines':
            alfa1, r1, s11, s12, color1, cat1 = extract_from(pieces[idx1]['extracted_lines'])
            alfa2, r2, s21, s22, color2, cat2 = extract_from(pieces[idx2]['extracted_lines'])
            # if len(alfa1) == 0 and len(alfa2) == 0:
            #     #print('no lines')
            #     #compatibility_matrix = np.zeros((m.shape[1], m.shape[1], len(rot))) + ppars.max_dist # this will be COMPATIBILITY - not COST !!!
            #     compatibility_matrix = np.zeros((m.shape[1], m.shape[1], len(rot))) + (ppars.badmatch_penalty/3)  # badmatch_penalty will be best compatibility
            # elif (len(alfa1) > 0 and len(alfa2) == 0) or (len(alfa1) == 0 and len(alfa2) > 0):
            #     #print('only one side with lines')
            #     #compatibility_matrix = np.zeros((m.shape[1], m.shape[1], len(rot))) + ppars.mismatch_penalty
            #     compatibility_matrix = np.zeros((m.shape[1], m.shape[1], len(rot)))
            # else:
            if len(alfa1) > 0 and len(alfa2) > 0:
                #print('values!')
                t1 = time.time()
                if compatibility_cost == 'DEBUG':
                    print(f"Computing compatibility between Piece {idx1} and Piece {idx2}")
                    if idx2 - idx1 == 1:
                        plt.suptitle(f"COST between Piece {idx1} and Piece {idx2}", fontsize=32)
                        compatibility_matrix = compute_cost_matrix_LAP_debug(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, color1, color2, cat1, cat2,
                                                    mask_ij, ppars, verbosity=verbosity, show=True)
                    else:
                        compatibility_matrix = compute_cost_matrix_LAP_debug(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, color1, color2, cat1, cat2, 
                                                    mask_ij, ppars, verbosity=verbosity, show=False)
                elif compatibility_cost == 'LAP':
                    compatibility_matrix = compute_line_based_CM_LAP(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, color1, color2, cat1, cat2, 
                                                    mask_ij, ppars, verbosity=verbosity)
                elif compatibility_cost == 'LCI':
                    compatibility_matrix = compute_line_based_CM_LCI(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, color1, color2, cat1, cat2, 
                                                            mask_ij, ppars, verbosity=verbosity)
                elif compatibility_cost == 'LAPvis':
                    lines_pi = alfa1, r1, s11, s12, color1, cat1
                    lines_pj = alfa2, r2, s21, s22, color2, cat2 
                    piece_i = pieces[idx1]
                    piece_j = pieces[idx2]
                    compatibility_matrix = compute_cost_matrix_LAP_vis(z_id, rot, lines_pi, lines_pj, piece_i, piece_j, mask_ij, ppars, verbosity=1)



                else:
                    print('weird: using {compatibility_cost} method, not known! We use `new` as we dont know what else to do! change --cmp_cost')
                    compatibility_matrix = compute_cost_matrix_LCI_method(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, color1, color2, cat1, cat2, 
                                                            mask_ij, ppars)
                if verbosity > 1:
                    print(f"computed cost matrix for piece {idx1} ({len(alfa1)} lines) vs piece {idx2} ({len(alfa2)} lines): took {(time.time()-t1):.02f} seconds ({candidate_values:.1f} candidate values) ")
                #print(compatibility_matrix)
        elif compatibility_type == 'shape':
            #breakpoint()
            ids_to_score = np.where(mask_ij > 0)
            compatibility_matrix = compute_SDF_CM_matrix(pieces[idx1], pieces[idx2], ids_to_score, ppars, verbosity=verbosity)
            #breakpoint()
        elif compatibility_type == 'motifs':
            assert ( (motif_det_method == "yolo-obb") | (motif_det_method == "yolo-bbox")), f"Unkown detection method for motifs!\nWe know `yolo-obb` and `yolo-bbox`, given `{motif_det_method}`\nRe-run specifying `--det_method`"
            compatibility_matrix = compute_CM_using_motifs(idx1, idx2, pieces, mask_ij, ppars, yolo_obj_detector=detector, det_type=motif_det_method, verbosity=verbosity)
        elif compatibility_type == 'color':
            compatibility_matrix = compute_cost_using_color_compatibility(idx1, idx2, pieces, mask_ij, ppars, seg_len=seg_len, verbosity=1)

        elif compatibility_type == 'Oracle_GT':
            compatibility_matrix = compute_oracle_compatibility(idx1, idx2, pieces, mask_ij, ppars, puzzle_root_folder, verbosity=1)

        else: # other compatibilities!
            print("\n" * 20)
            print("=" * 50)
            print("WARNING:")
            print(f"Received: {compatibility_type} as compatibility_type")
            print("NOT IMPLEMENTED YET, RETURNING JUST EMPTY MATRIX")
            print("=" * 50)
            print("\n" * 20)

            compatibility_matrix = np.zeros((m.shape[1], m.shape[1], len(rot)))
        
    return compatibility_matrix

def normalize_CM(R, parameters=None, region_mask=None):
    """
    It normalizes a compatibility matrix with a known structure (-1, 0, positive values) 
    """
    if not parameters or 'cmp_type' not in parameters.keys(): # standard
        R = np.maximum(-1, R)
        prm = (R > 0).astype(int)
        max_val = np.max(R[R > 0])
        scaling_factor = np.ones_like(R) * prm * max_val
        # R /= scaling_factor
        scaling_factor2 = scaling_factor + (1 - prm)
        normalized_R = R/scaling_factor2

    else:
        negative_region = np.minimum(region_mask, 0)
        if parameters['cmp_type'] == 'lines':
            if parameters['cmp_cost'] == 'LCI':
                #breakpoint()
                # TODO:
                normalized_R = R / np.max(R) # values between 0 and positive (length of pieces)

            elif parameters['cmp_cost'] == 'LAP':
                normalized_R = R / np.max(R) # values between 0 and parameters.badmatch_penalty
            else:
                print(f"What are you doing? Unknown cost: {parameters['cmp_cost']}")
                normalized_R = R

        elif parameters['cmp_type'] == 'color':
            # normalization
            k = parameters['k']
            R_cut = np.zeros((R.shape))
            a_ks = np.zeros((region_mask.shape[0], region_mask.shape[1], n))
            a_min = np.zeros((region_mask.shape[0], region_mask.shape[1], n))
            for i in range(n):
                a_cost_i = R[:, :, :, :, i]
                for x in range(a_cost_i.shape[0]):
                    for y in range(a_cost_i.shape[1]):
                        a_xy = a_cost_i[x, y, :, :]
                        a_all = np.array(np.unique(a_xy))
                        a = a_all[np.minimum(k, len(a_all) - 1)]
                        a_xy = np.where(a_xy > a, -1, a_xy)
                        a_cost_i[x, y, :, :] = a_xy
                        a_ks[x, y, i] = a
                        if len(a_all) > 1:
                            a_min[x, y, i] = a_all[1]
                print(a_ks[:, :, i])
                R_cut[:, :, :, :, i] = a_cost_i

            norm_term = np.max(a_ks) / (2 * k)
            normalized_R = 2 - R_cut / norm_term  # only for colors
            normalized_R = np.where(normalized_R > 2, 0, normalized_R)  # only for colors
            # normalized_R = np.where(normalized_R < 0, 0, normalized_R)   # only for colors
            normalized_R = np.where(normalized_R <= 0, -1, normalized_R)  ## NEW idea di Prof.Pelillo
            # normalized_R /= np.max(normalized_R)
        elif parameters['cmp_type'] == 'motifs':
            max_cost = np.max(R)
            if max_cost < 0.1:
                breakpoint()
            normalized_R = (np.clip(R, 0, max_cost)) / max_cost
        elif parameters['cmp_type'] == 'shape':
            normalized_R = R / np.max(R)
        else:
            print("\n\n### WARNING\nNo normalization used!\n\n")
            normalized_R = R
            negative_region = np.minimum(region_mask, 0)  # recover overlap (negative) areas

        normalized_R = normalized_R + negative_region  # insert negative regions to cost matrix
    return normalized_R

def reshape_list2mat(comp_as_list, n):
    first_element = comp_as_list[0]
    cost_matrix = np.zeros((first_element.shape[0], first_element.shape[1], first_element.shape[2], n, n))
    for i in range(n):
        for j in range(n):
            cost_matrix[:,:,:,j,i] = comp_as_list[i*n + j]
    return cost_matrix

def reshape_list2mat_and_normalize(comp_as_list, n, norm_value):
    """
    Old code, should not be used, 
    preferrable to use:
    - reshape_list2mat() and then 
    - normalize_CM() 
    """
    first_element = comp_as_list[0]
    cost_matrix = np.zeros((first_element.shape[0], first_element.shape[1], first_element.shape[2], n, n))
    norm_cost_matrix = np.zeros((first_element.shape[0], first_element.shape[1], first_element.shape[2], n, n))
    for i in range(n):
        for j in range(n):
            cost_matrix[:,:,:,j,i] = comp_as_list[i*n + j]
            norm_cost_matrix[:,:,:,j,i] = np.maximum(1 - cost_matrix[:,:,:,j,i] / norm_value, 0)
    return cost_matrix, norm_cost_matrix


def show_debug_visualization(pieces, i, j, args, R, region_mask, ppars):
    import matplotlib.pyplot as plt
    rotation_idx = 0
    plt.suptitle(f"CM: `{args.cmp_type}` (cost `{args.cmp_cost}`)", fontsize=45)
    plt.subplot(541); plt.imshow(pieces[i]['img']); plt.title(f"piece {i}"); plt.colorbar()
    plt.subplot(542); plt.imshow(pieces[j]['img']); plt.title(f"piece {j}"); plt.colorbar()
    plt.subplot(545); plt.imshow(region_mask[:,:,0,j,i], vmin=-1, vmax=1, cmap='RdYlGn'); plt.title("region map 0"); plt.colorbar()
    if region_mask.shape[2] > 1:
        plt.subplot(546); plt.imshow(region_mask[:,:,1,j,i], vmin=-1, vmax=1, cmap='RdYlGn'); plt.title("region map 1"); plt.colorbar()
        plt.subplot(547); plt.imshow(region_mask[:,:,2,j,i], vmin=-1, vmax=1, cmap='RdYlGn'); plt.title("region map 2"); plt.colorbar()
        plt.subplot(548); plt.imshow(region_mask[:,:,3,j,i], vmin=-1, vmax=1, cmap='RdYlGn'); plt.title("region map 3"); plt.colorbar()
    # plt.subplot(546); plt.imshow(R[:,:,rotation_idx], cmap='RdYlGn'); plt.title("cost")
    # if args.cmp_cost == 'LCI':
    #     norm_cmp = R[:,:,0] / np.max(R[:,:,0]) #np.maximum(1 - R[:,:,0] / parameters.rmax, 0)
    #     plt.subplot(547); plt.imshow(norm_cmp, vmin=-1, vmax=1, cmap='RdYlGn'); plt.title("compatibility")
    #     plt.subplot(548); plt.imshow(norm_cmp + np.minimum(region_mask[:,:,rotation_idx,i,j], 0), vmin=-1, vmax=1, cmap='RdYlGn'); plt.title("final cmp")
    # else:
    #     norm_cmp = np.maximum(1 - R[:,:,0] / parameters.rmax, 0)
    #     plt.subplot(547); plt.imshow(norm_cmp, vmin=-1, vmax=1, cmap='RdYlGn'); plt.title("compatibility")
    #     plt.subplot(548); plt.imshow(norm_cmp + np.minimum(region_mask[:,:,rotation_idx,i,j], 0), vmin=-1, vmax=1, cmap='RdYlGn'); plt.title("final cmp")
    
    plt.subplot(549); plt.title("COST ROTATION 0")
    plt.imshow(R[:,:,0], cmap='RdYlGn', vmin=-1, vmax=1); plt.colorbar() 
    if R.shape[2] > 1:
        plt.subplot(5,4,10); plt.title("COST ROTATION 1")
        plt.imshow(R[:,:,1], cmap='RdYlGn', vmin=-1, vmax=1); plt.colorbar() 
        plt.subplot(5,4,11); plt.title("COST ROTATION 2")
        plt.imshow(R[:,:,2], cmap='RdYlGn', vmin=-1, vmax=1); plt.colorbar() 
        plt.subplot(5,4,12); plt.title("COST ROTATION 3")
        plt.imshow(R[:,:,3], cmap='RdYlGn', vmin=-1, vmax=1); plt.colorbar() 

    if args.cmp_cost == 'LAP':
        R[R > ppars.badmatch_penalty] = ppars.badmatch_penalty
        ji_unique_values = np.unique(R)
        k = min(ppars.k, len(ji_unique_values))
        kmin_cut_val = np.sort(ji_unique_values)[-k]
        if kmin_cut_val == 0:
            kmin_cut_val = np.min(ji_unique_values[ji_unique_values > 0])
        plt.subplot(5,4,13); plt.title(f"COST KMINCUT ({kmin_cut_val:.2f}) ROTATION  0"); 
        plt.imshow(np.maximum(1 - R[:,:,0] / kmin_cut_val, 0), cmap='RdYlGn'); plt.colorbar()
        plt.subplot(5,4,14); plt.title(f"COST KMINCUT ({kmin_cut_val:.2f}) ROTATION  1")
        plt.imshow(np.maximum(1 - R[:,:,1] / kmin_cut_val, 0), cmap='RdYlGn', vmin=-1, vmax=1); plt.colorbar()
        plt.subplot(5,4,15); plt.title(f"COST KMINCUT ({kmin_cut_val:.2f}) ROTATION  2")
        plt.imshow(np.maximum(1 - R[:,:,2] / kmin_cut_val, 0), cmap='RdYlGn', vmin=-1, vmax=1) ; plt.colorbar()
        plt.subplot(5,4,16); plt.title(f"COST KMINCUT ({kmin_cut_val:.2f}) ROTATION  3")
        plt.imshow(np.maximum(1 - R[:,:,3] / kmin_cut_val, 0), cmap='RdYlGn', vmin=-1, vmax=1); plt.colorbar()
        plt.subplot(5,4,17); 
        plt.title("EXP ROTATION 0")
        sigma = 76 # why?
        plt.imshow(np.exp(-R[:,:,0]/sigma), cmap='RdYlGn'); plt.colorbar() 
        if R.shape[2] > 1:
            plt.subplot(5,4,18); plt.title("EXP ROTATION 1")
            plt.imshow(np.exp(-R[:,:,1]/sigma), cmap='RdYlGn', vmin=-1, vmax=1); plt.colorbar() 
            #plt.imshow(norm_cmp + np.minimum(region_mask[:,:,1,i,j], 0), vmin=-1, vmax=1, cmap='RdYlGn'); 
            plt.subplot(5,4,19); plt.title("EXP ROTATION 2")
            plt.imshow(np.exp(-R[:,:,2]/sigma), cmap='RdYlGn', vmin=-1, vmax=1); plt.colorbar() 
            #plt.imshow(norm_cmp + np.minimum(region_mask[:,:,2,i,j], 0), vmin=-1, vmax=1, cmap='RdYlGn'); 
            plt.subplot(5,4,20); plt.title("EXP ROTATION 3")
            plt.imshow(np.exp(-R[:,:,3]/sigma), cmap='RdYlGn', vmin=-1, vmax=1); plt.colorbar() 
        #plt.imshow(norm_cmp + np.minimum(region_mask[:,:,3,i,j], 0), vmin=-1, vmax=1, cmap='RdYlGn'); 
    
    if args.cmp_cost == 'LAP2':
        clipping_val = ppars.max_dist + (ppars.badmatch_penalty - ppars.max_dist) / 3
        R = np.clip(R, 0, clipping_val)
        R_normalized = 1 - R / clipping_val
        plt.subplot(5,4,13); plt.title("compatibility normalized")
        plt.imshow(R_normalized, cmap='RdYlGn'); plt.colorbar()
        plt.subplot(5,4,14); plt.title(f"COST KMINCUT ({kmin_cut_val:.2f}) ROTATION  1")
        plt.imshow(np.maximum(1 - R[:,:,1] / kmin_cut_val, 0), cmap='RdYlGn', vmin=-1, vmax=1); plt.colorbar()
        plt.subplot(5,4,15); plt.title(f"COST KMINCUT ({kmin_cut_val:.2f}) ROTATION  2")
        plt.imshow(np.maximum(1 - R[:,:,2] / kmin_cut_val, 0), cmap='RdYlGn', vmin=-1, vmax=1) ; plt.colorbar()
        plt.subplot(5,4,16); plt.title(f"COST KMINCUT ({kmin_cut_val:.2f}) ROTATION  3")
        plt.imshow(np.maximum(1 - R[:,:,3] / kmin_cut_val, 0), cmap='RdYlGn', vmin=-1, vmax=1); plt.colorbar()
        #plt.imshow(norm_cmp + np.minimum(region_mask[:,:,0,i,j], 0), vmin=-1, vmax=1, cmap='RdYlGn'); 
        plt.subplot(5,4,18); plt.title("EXP ROTATION 1")
        plt.imshow(np.exp(-R[:,:,1]/sigma), cmap='RdYlGn', vmin=-1, vmax=1); plt.colorbar() 
        #plt.imshow(norm_cmp + np.minimum(region_mask[:,:,1,i,j], 0), vmin=-1, vmax=1, cmap='RdYlGn'); 
        plt.subplot(5,4,19); plt.title("EXP ROTATION 2")
        plt.imshow(np.exp(-R[:,:,2]/sigma), cmap='RdYlGn', vmin=-1, vmax=1); plt.colorbar() 
        #plt.imshow(norm_cmp + np.minimum(region_mask[:,:,2,i,j], 0), vmin=-1, vmax=1, cmap='RdYlGn'); 
        plt.subplot(5,4,20); plt.title("EXP ROTATION 3")
        plt.imshow(np.exp(-R[:,:,3]/sigma), cmap='RdYlGn', vmin=-1, vmax=1); plt.colorbar() 
        #plt.imshow(norm_cmp + np.minimum(region_mask[:,:,3,i,j], 0), vmin=-1, vmax=1, cmap='RdYlGn'); 
    if args.cmp_cost == 'LAP2':
        clipping_val = ppars.max_dist + (ppars.badmatch_penalty - ppars.max_dist) / 3
        R = np.clip(R, 0, clipping_val)
        R_normalized = 1 - R / clipping_val
        plt.subplot(5,4,13); plt.title("compatibility normalized")
        plt.imshow(R_normalized, cmap='RdYlGn'); plt.colorbar()
    
    plt.show()
    breakpoint()
    return 0