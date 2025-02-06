"""
Function signature and parameters explanation:

def compute_CM_using_geom_good_continuation(idx1, idx2, pieces, mask_ij, ppars):

    :param int idx1: the index of the first piece (which will be in the center)
    :param int idx2: the index of the second piece (which will move around)
    :param list pieces: 
                a list of dict. pieces[idx1] gets the dict with all the information 
                related to the first piece.
                pieces[idx1] has the following keys:
                'img': the color image, 
                'mask': the binary mask (1 fragment, 0 background), 
                'cm': center of mass of the fragment, 
                'id': id of the fragment (useful mostly for repair), 
                'name': name of the fragment, 
                'polygon': shapely.Polygon with the shape of the fragment
            
                ** optionally, depending on dataset, we might load also:
                ['sdf', 'segmented_poly', 'boundary_seg'], but they are very specific, 
                not important for general purposes
    :param numpy.ndarray mask_ij: 
                is a mask (with shape x,y,z like the grid of possible points where 
                you place the second piece) which tells you the positions where 
                the score should be computed. 
                So if mask_ij[y, x, t] == 1 the score should be computed, 
                if mask_ij[y, x, t] == 0 the score can be left as a zero.
    :param dict ppars: 
                it's a dictionary with lots of parameters related to the puzzle.
                the keys are: 
                ['piece_size', 'num_pieces', 'img_size', 'p_hs', 'xy_step', 'xy_grid_points', 
                'theta_step', 'theta_grid_points', 'pairwise_comp_range', 'canvas_size', 
                'comp_matrix_shape', 'threshold_overlap', 'threshold_overlap_lines', 
                'threshold_overlap_motifs', 'borders_regions_width_outside', 
                'borders_regions_width_inside', 'border_tolerance', 'cmp_type', 
                'cmp_cost', 'det_method', 'calc_sdf', 'line_based', 'motif_based', 
                'color_based', 'seg_len', 'k', 'lines_det_method', 'motif_det_method', 
                'p', 'z_id', 'm', 'rot']

                some important ones are:
                - 'xy_grid_points': (int) is an integer number of positions considered. 
                        If the value is 11, it means we try to place the second piece around 
                        the first one in a virtual grid of 11x11 positions
                - 'xy_step': (int) the distance between consecutive grid points (in pixels)
                - 'theta_grid_points': (int) the number of rotations considered
                - 'theta_step': (float) the distance between rotations (in degrees)
                - 'num_pieces': (int) the number of pieces in this puzzle
                - 'piece_size': (width, height, channel) the tuple of the size of the image of each piece
        
    :return: a compatibility matrix (CM) of shape ([xy_grid_points, xy_grid_points, theta_grid_points]) - (is the same shape as `mask_ij`!) 
             containing the compatibility scores. 
             The scores can range from -1 to 1.
             A high score (--> 1) at position [y,x,t] means that an assembly which places the first piece in (0,0,0) and 
             the second piece in (y,x,t) is considered a goood assembly!
             A negative score (<0) at position [y,x,t] means that an assembly which places the first piece in (0,0,0) and 
             the second piece in (y,x,t) should not be considered by the solver.
             A zero score at position [y,x,t] means that an assembly which places the first piece in (0,0,0) and 
             the second piece in (y,x,t) is indifferent (the solver will use other information coming from other pairs of pieces to decide).
             (Note: zero score is not "bad", is just there is not enough information to decide based on this pair of pieces and position).
    :rtype: numpy.ndarray
"""