# Compatibility

Here we compute various compatibility measures. 

## Parameters

To have some consistency in the computation, we defined this set of *standard* parameters:

For the fragment ones
```
# Fragments (repair)
pairwise compatibility matrix size: 101 x 101 x 24 (x, y, theta)
piece size: 251 x 251 pixels
pairwise compatibility range: 2 x (piece size - 1) = 500 x 500 pixels
step between grid positions: 
    - on x and y: 500 / (101-1) = 5 pixels 
    - on theta: 360 degress / 24 = 15 degrees = 0.2617993878 rad
```

For the squared puzzle (generated from images)
``` 
# images
images resized to: 1204x1204
piece size: 301x301
number of pieces: 16
possible positions: 3x3x4 (3 on x and y axis, 4 rotations)
```