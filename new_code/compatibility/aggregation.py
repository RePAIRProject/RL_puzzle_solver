import scipy 
import numpy as np 
import cv2 
from utils.puzzle_utils import PuzzlePiece
import shapely 
from utils.puzzle_utils import Puzzle 
from utils.parameters_utils import Configuration, CustomYAMLEncoder
from compatibility.grid import PuzzleGrid, PieceOnCanvas
import yaml
# only for debug, they should not be used here
import os
import random


class AggregationModule:

    def __init__(self, puzzle: Puzzle, params: dict, cfg: Configuration):

        self.puzzle = puzzle
        self.input_params = params
        self.params = params['aggregation']
        self.method = self.params['method']
        self.cfg = cfg 
        
        self._init()

    def _init(self):
        """ Loads the data """
        self.CM_dict = np.load(self.cfg.get_CM_path(), allow_pickle=True).item()
        print("loaded the CM")
        for cmk in self.CM_dict.keys():
            if "__" not in cmk:
                print(f"Found {cmk}-based compatibility")
        self.grid_params = self.CM_dict['__context']['grid_params']

        self.RM_dict = np.load(self.cfg.get_RM_path(), allow_pickle=True).item()

    def compute(self, verbose:int = 1):
        """ The wrapper that computes the aggregation """
        if self.method == 'SLM':
            R = _aggregate_shape_lines_motives(self)
        elif self.method == 'SM':
            R = _aggregate_shape_motives(self)
        elif self.method == 'shape':
            R = self.CM_dict['shape']
        elif self.method == 'motif' or self.method == 'motives':
            R = self.CM_dict['motif']
        elif self.method == 'lines':
            R = self.CM_dict['lines']
        elif self.method == 'PAD' or self.method == 'pairwise_alignment_discriminator':
            R = self.CM_dict['pairwise_alignment_discriminator']
        elif self.method == 'oracle':
            R = self.CM_dict['oracle']
        else:
            R = self.CM_dict['shape']

        self.CM_dict['R'] = R
    

    def save(self):
        """ save the data in a .npy file, context parameters and values """

        # read the context values and add the aggregation ones
        context_params = self.CM_dict['__context']
        context_params['aggregation'] = self.params
        self.CM_dict['__context'] = context_params

        # save the file with all the matrices (in the same file, overwrite)
        np.save(self.cfg.get_CM_path(), self.CM_dict)
        
        # input parameters
        input_params_path = self.cfg.get_aggregation_input_parameters_path() 
        with open(input_params_path, 'w') as f:
            yaml.dump(self.input_params, f, Dumper=CustomYAMLEncoder, default_flow_style=False)
        # output parameters
        context_params_path = self.cfg.get_aggregation_output_parameters_path() 
        with open(context_params_path, 'w') as f:
            yaml.dump(context_params, f, Dumper=CustomYAMLEncoder, default_flow_style=False)
        print("done, saved the CM.")

    def _aggregate_shape_motives(self):
        return 1

    def _aggregate_shape_lines_motives(self, lines_avg_val: float = 0.5, motif_avg_val: float = 0.5):
        """ It combines three compatibilities (ShapeLinesMotifs) """

        R_shape = self.CM_dict['shape']
        R_lines = self.CM_dict['lines']
        R_motif = self.CM_dict['motif']

        RM_lines = self.RM_dict['lines']
        RM_motif = self.RM_dict['motif']
        RM_shape = self.RM_dict['shape']

        norm_R_shape = AggregationModule._normalize_shape_based_CM(R_shape)
        norm_R_lines = AggregationModule._normalize_line_based_CM(R_lines)
        norm_R_motif = AggregationModule._normalize_motif_based_CM(R_motif)

        negative_region_map = R_shape < 0
        region_motif = AggregationModule._combine_RMs([RM_shape, RM_motif])
        region_lines = AggregationModule._combine_RMs([RM_shape, RM_lines])

        prm_motif = (region_motif > 0).astype(int)  ## positive in RM
        prm_lines = (region_lines > 0).astype(int)  ## positive in RM
        prm_shape = (shape_RM > 0).astype(int)
        shape_basis = norm_R_shape * prm_shape

        # lines_avg_val = 0.5  # fix level ???  # lines_avg_val1 = np.mean(norm_R_lines > 0)
        # motif_avg_val = 0.5  # motif_avg_val1 = np.mean(norm_R_motif > 0)
        motif_contrib = prm_motif * ((norm_R_motif / motif_avg_val) - 1)
        lines_contrib = prm_lines * ((norm_R_lines / lines_avg_val) - 1)

        R = np.zeros_like(R_shape)
        total_contrib = shape_basis * (motif_contrib + lines_contrib)
        R = shape_basis + total_contrib
        R += -1 * negative_region_map.astype(int)
        R = _normalize_CM(R)
        R = np.maximum(-1, R)
        return R

    def _normalize_CM_wrapper(self, CM: np.ndarray, CM_type: str = 'unknown', RM: np.ndarray = None):
    # def normalize_CM(R, parameters=None, region_mask=None):
        """
        It normalizes a compatibility matrix with a known structure (-1, 0, positive values) 
        """
        if RM is not None:
            negative_region = np.minimum(RM, 0)
        else:
            negative_region = np.zeros_like(CM)

        if CM_type == 'lines':
            normalized_CM = AggregationModule._normalize_line_based_CM(CM)
        elif CM_type == 'color':
            normalized_CM = AggregationModule._normalize_color_based_CM(CM)
        elif CM_type == 'motif':
            normalized_CM = AggregationModule._normalize_motif_based_CM(CM)
        elif CM_type == 'shape':
            normalized_CM = AggregationModule._normalize_shape_based_CM(CM)
        else:
            normalized_CM = AggregationModule._normalize_CM(CM)

        normalized_R = normalized_CM + negative_region  # insert negative regions to cost matrix
       
        return normalized_R
    
    @staticmethod
    def _normalize_line_based_CM(CM: np.ndarray):
        if parameters['cmp_cost'] == 'LCI':
            # TODO:
            normalized_CM = CM / np.max(CM) # values between 0 and positive (length of pieces)
        elif parameters['cmp_cost'] == 'LAP':
            normalized_CM = CM / np.max(CM) # values between 0 and parameters.badmatch_penalty
        else:
            print(f"What are you doing? Unknown cost: {parameters['cmp_cost']}")
            normalized_CM = CM
        return normalized_CM

    @staticmethod
    def _normalize_color_based_CM(CM: np.ndarray, k: int = 10):
        # k = parameters['k']
        R_cut = np.zeros((CM.shape))
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
        normalized_CM = 2 - R_cut / norm_term  # only for colors
        normalized_CM = np.where(normalized_CM > 2, 0, normalized_CM)  # only for colors
        # normalized_R = np.where(normalized_R < 0, 0, normalized_R)   # only for colors
        normalized_CM = np.where(normalized_CM <= 0, -1, normalized_CM)  ## NEW idea di Prof.Pelillo
        return normalized_CM

    @staticmethod
    def _normalize_motif_based_CM(CM: np.ndarray):
        max_cost = np.max(CM)
        if max_cost < 0.1:
            breakpoint()
        normalized_CM = (np.clip(CM, 0, max_cost)) / max_cost
        return normalized_CM

    @staticmethod
    def _normalize_shape_based_CM(CM: np.ndarray):
        normalized_CM = CM / np.max(CM)
        return normalized_CM

    @staticmethod
    def _normalize_CM(CM: np.ndarray):
        CM = np.maximum(-1, CM)
        prm = (CM > 0).astype(int)
        max_val = np.max(R[R > 0])
        scaling_factor = np.ones_like(CM) * prm * max_val
        # R /= scaling_factor
        scaling_factor2 = scaling_factor + (1 - prm)
        normalized_CM = CM / scaling_factor2
        return normalized_CM

    @staticmethod
    def _combine_RMs(RMs: list):
        neg_reg = RMs[0] < 0
        combined_pos = RMs[0] * (RMs[0] > 0).astype(int)
        for i in range(1, len(RMs)):
            combined_pos *= RMs[i] * (RMs[i] > 0).astype(int)
        combined = combined_pos - neg_reg
        return combined