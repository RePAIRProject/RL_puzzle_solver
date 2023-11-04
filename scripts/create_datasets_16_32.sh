

# to be sure path is correct
export PYTHONPATH=$PWD:$PYTHONPATH

# 16x16, 50 lines
python datasets/create_line_dataset_16x16.py 50 lines

# 16x16, 50 segments
python datasets/create_line_dataset_16x16.py 50 segments

# 32x32, 50 lines
python datasets/create_line_dataset_32x32.py 50 lines

# 32x32, 50 segments
python datasets/create_line_dataset_32x32.py 50 segments

# 32x32, 75 lines
python datasets/create_line_dataset_32x32.py 75 lines

# 32x32, 75 segments
python datasets/create_line_dataset_32x32.py 75 segments

# 32x32, 100 lines
python datasets/create_line_dataset_32x32.py 100 lines

# 32x32, 100 segments
python datasets/create_line_dataset_32x32.py 100 segments