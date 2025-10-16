import math
import numpy as np

def normalize_solutions(solutions, center, reference_frag):
    import math
    # Reference fragment
    x_ref, y_ref, t_ref = solutions[reference_frag]

    # Convert rotation angle from degrees to radians
    theta = math.radians(-t_ref)  # Negate for inverse rotation
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    transformed = []     #sol_lists = solutions.tolist()
    for x, y, t in solutions:
        # Step 1: Translate (bring the reference to origin)
        x_shifted = x - x_ref + center[0]
        y_shifted = y - y_ref + center[1]
        # Step 2: Rotate around origin (0,0) using inverse rotation of reference
        x_rotated = x_shifted * cos_theta - y_shifted * sin_theta
        y_rotated = x_shifted * sin_theta + y_shifted * cos_theta
        # Step 3: Adjust rotation angle
        t_rotated = t - t_ref
        transformed.append((x_rotated, y_rotated, t_rotated))
        #transformed = np.array(transformed, dtype=np.int64)

    return transformed


def probability_for_single_fragment(grid_size, mean, std_devs):

    # mean = (25.312, 25, 90.2587)  ## solution for the piece, t° !
    # std_devs = (10.0, 10.0, 0.5)  ## st. deviation, t° !

    size_x, size_y, size_theta = grid_size
    cx, cy, ct = mean
    sx, sy, st = std_devs
    ct = ct/ 360 * grid_size[2]  ### conversion to "cycle-grid"
    st = st/ 360 * grid_size[2]  ### conversion to "cycle-grid"

    # Create 3D grid of coordinates
    x = np.arange(size_x)
    y = np.arange(size_y)
    t = np.arange(size_theta)
    X, Y, T = np.meshgrid(x, y, t, indexing='ij')

    # Compute squared distances
    dx2 = ((X - cx) ** 2) / (2 * sx ** 2)
    dy2 = ((Y - cy) ** 2) / (2 * sy ** 2)

    # Cyclic angular distance
    dtheta = np.minimum(np.abs(T - ct), size_theta - np.abs(T - ct))
    #    np.minimum(np.abs(a - b), cycle_length - np.abs(a - b))   #"""Compute minimum cyclic distance between a and b."""

    dt2 = (dtheta ** 2) / (2 * st ** 2)
    prob = np.exp(-(dx2 + dy2 + dt2))
    prob /= np.sum(prob)

    return prob


####################################
####################################
import math
import numpy as np

grid_size = (191, 191, 8)     ## p_size
center = np.array([grid_size[0]//2, grid_size[1]//2], dtype=np.int64)

solutions = [
    (10, 5, 90),   # reference_frag
    (12, 8, 0),
    (15, 6, 180),
]
solutions = np.array(solutions, dtype=np.int64)
reference_frag = 0

normalized = normalize_solutions(solutions, center, reference_frag)
normalized[:,:2] = normalized[:,:2] + center
normalized[:, 2] = (normalized[:,2]+360)%360

p = np.zeros((grid_size[0], grid_size[1], grid_size[2], len(solutions[0])))

for i in range(len(normalized)):
    mean = normalized[i, :]  ## solution for the piece, t° !
    #std_devs = variance[i, :]
    std_devs = (100.0, 100.0, 50)  ## st. deviation, t° !
    prob = probability_for_single_fragment(grid_size, mean, std_devs)
    p[:, :, :, i] = prob


########################################################
# Show distribution for (theta = 0, 1, ... , n_of_slice)
import matplotlib.pyplot as plt
vis = 1
for i in range(len(solutions)):
    prob_i = p[:, :, :, i]

    if vis == 1:
        # Show distribution for (theta = 0, 1, ... , n_of_slice)
        n_of_slice = 4
        vmin = prob_i.min()  # global limits of the scale
        vmax = prob_i.max()
        fig, axes = plt.subplots(1, n_of_slice, figsize=(30, 10))

        for j in range(n_of_slice):
            ax = axes[j]  # map 0–5 in (row, col)
            im = ax.imshow(prob_i[:, :, j], cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
            ax.set_title(f"θ = {j} fragment{i} anc {anchor_idx}")
            ax.set_xlabel("y")
            ax.set_ylabel("x")
        # colorbar comune a tutti i subplot
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.07,
                            fraction=0.05, )  # shrink=0.8, orientation='horizontal',fraction=0.05,
        cbar.set_label("Probability Density")
        plt.suptitle("distribution for different rotations θ (Uniform Colors)", fontsize=18)
        plt.show()

