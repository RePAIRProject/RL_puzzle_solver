from utils.parameters_utils import Configuration
from solver.solver import SolverModule

def main():

    cfg = Configuration() # this contains all IO operations plus the folder structure
    params = cfg.load('input_parameters_local.yaml')   
    # we are solving based on the CM previously computed, so we need to know `exp_name`
    cfg.set_puzzle_single_run_random_folder_name('exp_kylbol')

    sm = SolverModule(params, cfg)
    sm.solve(verbose=params['verbosity'])
    sm.save()

if __name__ == '__main__':
    main()


