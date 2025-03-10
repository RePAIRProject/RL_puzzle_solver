# CODE STRUCTURE 

```bash
.
├── data/                           # the raw data before preprocessing
│   └── repair_g3                   # example, uncentered/unprocessed images of the pieces
│                               
├── preprocessing/                  # output from preprocessing
│   │   ├── images                  # the images of single pieces (created)
│   │   ├── masks                   # the binary masks of single pieces (created)
│   │   └── polygons                # the Shapely polygons of single pieces (created)
│   └── preprocessing.yaml          # the parameters used for preprocessing
│
├── features/                       # the features extracted 
│   │   ├── lines                   # lines 
│   │   ├── motifs                  # motifs
│   │   └── HSI                     # hyperspectral 
│   └── features.yaml               # the parameters used for extracting them
│
├── experiments/                    # experiments (running RM, CM, Solver)
│   ├── exp_random                  # each single experiment has a folder with exp_{random string} as a name
│   │   ├── experiment.yaml         # configuration file with input parameters (copy of input_parameters.yaml)
│   │   ├── history.yaml            # configuration file with output parameters (computed)
│   │   ├── regions/                # region matrix folder 
│   │   │   ├── visualization       # visualization    
│   │   │   └── RM.mat/npy          # file with values of RM
│   │   ├── compatibilities/        # lines 
│   │   │   ├── visualization       # visualization    
│   │   │   └── R.mat/npy           # file with values of R
│   │   ├── solution/        
│   │   │   ├── visualization       # visualization    
│   │   │   └── P.mat/npy           # file with final values of P   
│   │   └── metrics/   
│   │       └── evaluation.json     # numerical evaluation (when available)   
│   ├── exp_rand02/..
│   └── exp_rand03/..
│                                   # CODE-related part
├── input_parameters.yaml           # ALL (?) input parameters (including preproc? and features?)
├── default.yaml                    # default values (when they are not set, take these!)
├── preprocess_data.py              # wrapper for preprocessing 
├── extract_features.py             # wrapper for extracting "general" features (due to the input parameters)
├── compute_region_matrix.py        # wrapper for region matrices
├── compute_compatibility_matrix.py # wrapper for compatibility matrices   
├── run_solver.py                   # wrapper for solving
├── evaluate_solutions.py           # wrapper for computing all metrics
├── utils/ 
│   ├── preprocessing_utils.py      # here the actual code for each kind of preprocessing
│   ├── features_utils.py           # here the actual code for each kind of feature
│   ├── regions_utils.py            # defines the RM class and the code ?
│   ├── compatibility_utils.py      # defines the R/CM class and the code ?
│   ├── solver_utils.py             # defines the Solver class and the code ? 
│   ├── evaluation_utils.py         # for the different metrics
│   └── pieces_utils.py             # read and prepare the pieces - defines the Pieces class and the code ?
│
└── README.md                       # documentation
```
