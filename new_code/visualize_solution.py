import numpy as np
from scipy.io import savemat
import argparse
import pdb
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import json, os

from compatibility.region import RegionMatrixModule
from compatibility.compatibility import CompatibilityMatrixModule
from compatibility.aggregation import AggregationModule
from solver.solver import SolverModule
from utils.puzzle_utils import Puzzle
from utils.parameters_utils import Configuration

def main():

    cfg = Configuration() # this contains all IO operations plus the folder structure

    # Read solution
    np.loadtxt(self.cfg.get_solution_as_csv_path())
    # Load pieces
    # Create image



if __name__ == '__main__':

    main()