```bash
.
├── data/                                       # info ?
│   ├── dataset                                 # the raw data before preprocessing
│   │   ├── repair                              # dataset
│   │   │   ├── g1                              # all pieces, maybe in groups
│   │   │   └── g2                              # maybe all together?
│   │   └── dafne                               # or any other dataset 
│   │       └── p1                              # puzzle, uncentered/unprocessed images of the pieces
│   │                                       
│   ├── preprocessing/                          # output from preprocessing
│   │   ├── repair_g3                           # name combined dataset_puzzle
│   │   │   ├── images                          # the images of single pieces (created)
│   │   │   ├── masks                           # the binary masks of single pieces (created)
│   │   │   ├── polygons                        # the Shapely polygons of single pieces (created)
│   │   │   └── preprocessing.yaml              # the parameters used for preprocessing (copied/created)
│   │   ├── repair_mix_name                     # OR name of puzzle created from mix of pieces
│   │   │   ├── images                          # the images of single pieces (created)
│   │   │   ├── masks                           # the binary masks of single pieces (created)
│   │   │   ├── polygons                        # the Shapely polygons of single pieces (created)
│   │   │   └── preprocessing.yaml              # the parameters used for preprocessing (copied/created)
│   │   └── preprocessing_default.yaml          # the default parameters for preprocessing
│   │       
│   ├── features/                               # the features extracted 
│   │   ├── repair_g3                           # name combined dataset_puzzle
│   │   │   ├── lines                           # lines 
│   │   │   ├── motifs                          # motifs
│   │   │   ├── HSI                             # hyperspectral 
│   │   │   └── features.yaml                   # the parameters used for extraction (copied/created) 
│   │   ├── repair_mix_name                     # OR name of puzzle created from mix of pieces
│   │   │   ├── lines                           # lines 
│   │   │   ├── motifs                          # motifs
│   │   │   ├── HSI                             # hyperspectral 
│   │   │   └── features.yaml                   # the parameters used for extraction (copied/created) 
│   │   └── features_default.yaml               # the default parameters for extracting features
│   │   
│   └── puzzles/                                # experiments (running RM, CM, Solver)
│       └── repair_mix_name                     # name of the puzzle (can be a group, can be a mix)
│           ├── {timestamp}_rand01              # each single experiment has a folder with exp_{random string} as a name
│           │   ├── experiment.yaml             # configuration file with input parameters (copy of input_parameters.yaml)
│           │   ├── history.yaml                # configuration file with output parameters (computed)
│           │   │   
│           │   ├── compatibilities/            # EVERYTHING ABOUT COMPATIBILITIES 
│           │   │   ├── scores/                 # CM folder
│           │   │   │   ├── visualization       # visualization folder    
│           │   │   │   ├── R_shape.npy         # file with values of R
│           │   │   │   ├── R_lines.npy         # file with values of R
│           │   │   │   ├── ..                  # file with values of R
│           │   │   │   ├── R_motives.npy       # file with values of R
│           │   │   │   └── params.yaml         # parameters used to create R!
│           │   │   │       
│           │   │   ├── regions/                # RM folder 
│           │   │   │   ├── visualization       # visualization folder    
│           │   │   │   ├── RM_shape.npy        # file with values of RM
│           │   │   │   ├── RM_lines.npy        # file with values of RM
│           │   │   │   ├── ..                  # file with values of RM
│           │   │   │   ├── RM_motives.npy      # file with values of RM
│           │   │   │   └── params.yaml         # parameters used to create RM!
│           │   │   │       
│           │   │   └── aggregated/             # Aggregation folder 
│           │   │       ├── visualization       # visualization folder    
│           │   │       ├── R_agg1.npy          # file with values of aggregated R
│           │   │       ├── R_agg2.npy          # file with values of aggregated R
│           │   │       ├── ..                  # file with values of aggregated R
│           │   │       ├── R_aggN.npy          # file with values of aggregated R
│           │   │       └── params.yaml         # parameters used to aggregate R!
│           │   │           
│           │   └── solutions/                  # SOLVER: different runs in different folders
│           │       ├── solution_{timestamp}    # folder of one run with certain parameters
│           │       │   ├── visualization       # visualization folder    
│           │       │   ├── P.npy               # file with final values of P       
│           │       │   └── params.yaml         # parameters used to solve
│           │       ├── solution_{timestamp}    # region matrix folder 
│           │       │   ├── visualization       # visualization folder  
│           │       │   ├── P.npy               # file with final values of P       
│           │       │   └── params.yaml         # parameters used to solve
│           │       └── metrics/    
│           │           └── evaluation.json     # numerical evaluation (when available)   
│           ├── {timestamp}_rand02/..   
│           └── {timestamp}_rand03/..   
│                                               # CODE-related part
├── input_parameters.yaml                       # ALL (?) input parameters (including preproc? and features?)
├── default.yaml                                # default values (when they are not set, take these!)
├── preprocess_data.py                          # wrapper for preprocessing 
├── extract_features.py                         # wrapper for extracting "general" features (due to the input parameters)
├── compute_region_matrix.py                    # wrapper for region matrices
├── compute_compatibility_matrix.py             # wrapper for compatibility matrices   
├── solve.py                                    # wrapper for solving
├── evaluate_solutions.py                       # wrapper for computing all metrics
├── run_complete_pipeline.py                    # wrapper for computing everything everywhere all at once
├── GUI/        
│   ├── backend/        
│   ├── frontend/       
│   └── cache/      
├── API/                                        # submodule ? 
├── preprocessing/      
│   ├── preprocessing.py                        # here the actual code for each kind of preprocessing
│   └── data_generation.py                      # cutting images? maybe separated 
├── features_extraction/        
│   └── features_extraction.py        # here     the actual code for each kind of feature
├── compatibility/      
│   ├── compatibility.py                        # compatibility module
│   ├── regions.py                              # region matrix module  
│   └── grid.py                                 # grid module
├── solver/     
│   └── solver.py                               # defines the Solver class and the code ? 
├── metrics/        
│   └── evaluations.py                          # for the different metrics
├── utils/      
│   ├── parameters.py                           # for the code to load/write/merge .yaml files and paths
│   └── puzzle.py                               # read and prepare the pieces - defines the PuzzlePiece class and the Puzzle class
│       
└── README.md                                   # documentation