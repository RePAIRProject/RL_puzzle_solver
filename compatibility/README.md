# Compatibility

Here we compute various compatibility measures. 

## Parameters

To have some consistency in the computation, we defined this set of *standard* parameters:

```
pairwise compatibility matrix size: 101 x 101 x 24 (x, y, theta)
piece size: 251 x 251 pixels
pairwise compatibility range: 2 x (piece size - 1) = 500 x 500 pixels
step between grid positions: 
    - on x and y: 500 / (101-1) = 5 pixels 
    - on theta: 360 degress / 24 = 15 degrees = 0.2617993878 rad
```