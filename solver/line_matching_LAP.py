
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.optimize import linear_sum_assignment
import scipy.io


def read_info(image):
    # read json
    fname = f'C:\\Users\\Marina\\Toy_Puzzle_Matlab\\precomputed/RPf_00{image}.json'
    with open(fname, 'r') as file:
        data = json.load(file)
    beta = np.array(data['angles'])
    R = np.array(data['dists'])
    # # read image
    # I = Image.open(
    # f'C:\\Users\\Marina\\PycharmProjects\\WP3-PuzzleSolving\\
    # Compatibility\\data\\repair\\group_28\\ready\\RPf_00{image}.png')
    return beta, R  # I


def translation2(beta, radius, point):
    # a*x + b*y + c = 0  - initial equation of the line
    a = np.cos(beta)
    b = np.sin(beta)
    c = -radius

    if b == 0:  # if beta 90°
        y = np.linspace(-2000, 5000)
        x = -c/a * np.ones_like(y)
    else:
        x = np.linspace(-2000, 5000)
        y = 1/b * (-a*x - c)

    # a*(x+point(x)) + b*(y+point(y)) + c = 0 -  equation of translated line
    c_new = (a*point[0] + b*point[1] + c)
    if b == 0:
        y_new = np.linspace(-2000, 5000)
        x = -c_new/a * np.ones_like(y)
    else:
        y_new = 1/b * (-a*x - c_new)
    R_new = -c_new
    # if or (and((1/b*(-a*p(1)-c)>p(2)), (1/b*(-a*0-c)>0)), and((1/b*(-a*p(1)-c)<p(2)), (1/b*(-a*0-c)<0)))
    if ((1/b*(-a*p[0]-c) > p[1]) and (1/b*(-a*0-c) > 0)) or ((1/b*(-a*p[0]-c) < p[1]) and (1/b*(-a*0-c) < 0)):
        beta_new = beta
    else:
        R_new = -R_new
        beta_new = (beta + np.pi) % (2*np.pi)

    return beta_new, R_new, x, y, y_new


def dist_point_line(beta, radius, point):
    a = np.cos(beta)
    b = np.sin(beta)
    c = -radius
    R_new = abs(a*point[0] + b*point[1] + c)/np.sqrt(a**2 + b**2)
    return R_new


# MAIN
rmax = 50
tr_coef = 0.1
max_dist = 1000
ang = 45
step = 0.04

m_size = 51
m = np.zeros((m_size, m_size, 2))
m2, m1 = np.meshgrid(np.linspace(-1, 1, m_size), np.linspace(-1, 1, m_size))
m[:, :, 0] = m1
m[:, :, 1] = m2

p = [500, 500]   # center of grid (0,0)
z_rad = 1000     # size of grid
z_id = m*z_rad
grid_step = z_id[1, 0, 0]-z_id[0, 0, 0]
rot = np.arange(0, 360-ang+1, ang)

pieces = np.concatenate((np.arange(194, 199), np.arange(200, 204)))
n = len(pieces)
All_cost = np.zeros((m.shape[1], m.shape[1], len(rot), len(pieces), len(pieces)))
All_cost2 = All_cost
Nnorm_cost = All_cost

for f1 in range(n):         # - select fixed fragment
    im1 = pieces[f1]        # read image 1
    alfa1, R1 = read_info(im1)
    for f2 in range(n):     # - select moving and rotating fragment
        im2 = pieces[f2]    # read image 2
        alfa2, R2 = read_info(im2)
        R_cost = np.zeros((m.shape[1], m.shape[1], len(rot)))
        R_cost2 = R_cost
        if f1 == f2:
            R_norm = R_cost - 1
            R_norm2 = R_norm
        else:
            for t in range(len(rot)):
                teta = -rot[t] * np.pi / 180    # rotation of F2
                for ix in range(m_size):        # (z_id.shape[0]):
                    for iy in range(m_size):    # (z_id.shape[0]):
                        z = z_id[ix, iy]
                        n_lines_f1 = alfa1.shape[0]
                        n_lines_f2 = alfa2.shape[0]
                        cost_matrix = np.zeros((n_lines_f1, n_lines_f2))

                        for i in range(n_lines_f1):
                            for j in range(n_lines_f2):
                                # translate reference point to the center
                                beta1, R_new1, x1, y1, y_new1 = translation2(alfa1[i], R1[i], p)
                                beta2, R_new2, x2, y2, y_new2 = translation2(alfa2[j], R2[j], p)
                                # shift and rot line 2
                                beta3, R_new3, x3, y3, y_new3 = translation2(beta2 + teta, R_new2, -z)

                                # dist from new point to line 1
                                R_new4 = dist_point_line(beta1, R_new1, z)

                                # distance between 2 lines
                                gamma = beta1 - beta3
                                coef = np.abs(np.sin(gamma))

                                dist1 = np.sqrt((R_new1**2 + R_new3**2 - 2*np.abs(R_new1*R_new3)*np.cos(gamma)))
                                dist2 = np.sqrt((R_new2**2 + R_new4**2 - 2*np.abs(R_new2*R_new4)*np.cos(gamma)))

                                if coef < tr_coef:
                                    cost = (dist1 + dist2)
                                else:
                                    cost = max_dist
                                cost_matrix[i, j] = cost
                                # end
                            # end
                        # end

                        # LAP
                        row_ind, col_ind = linear_sum_assignment(cost_matrix)  # [assig,tot_cost]=munkres(cost_matrix)
                        tot_cost = cost_matrix[row_ind, col_ind].sum()
                        R_cost[iy, ix, t] = tot_cost
                        # option MIN
                        r_c = cost_matrix[row_ind, col_ind].min()
                        R_cost2[iy, ix, t] = r_c
                    # end
                # end
            # end
            R_norm = np.maximum(1 - R_cost/rmax, 0)
            R_norm2 = np.maximum(1 - R_cost2/rmax, 0)
        # end
        Nnorm_cost[:, :, :, f2, f1] = R_cost
        All_cost[:, :, :, f2, f1] = R_norm
        All_cost2[:, :, :, f2, f1] = R_norm2
    # end
# end

Rmat_in_file = 'C:\\Users\\Marina\\Toy_Puzzle_Matlab\\R_mask51_45cont2.mat'

mat = scipy.io.loadmat('C:\\Users\\Marina\\Toy_Puzzle_Matlab\\R_mask51_45cont2.mat')
R_mask = mat['R_mask']
rneg = np.where(R_mask < 0, -1, 0)

R_line = All_cost * R_mask + rneg
R_line2 = All_cost2 * R_mask + rneg
for jj in range(n):
    R_line[:, :, :, jj, jj] = -1

R_line = R_line * 2
R_line[R_line < 0] = -0.5
for jj in range(n):
    R_line2[:, :, :, jj, jj] = -1
R_line2 = R_line2 * 2
R_line2[R_line2 < 0] = -0.5

np.save('R_line51_45_verLAP_fake2.npy', R_line2)
np.save('R_line51_45_verLAP_fakeNEW.npy', R_line)  # full LAP

#########################################################
# Visualize Rij matrices
jj = 0         # selected rotation

# Visualize matrices of cost before norm
plt.figure()
fig, axs = plt.subplots(n, n)
fig.subplots_adjust(hspace=0.1, wspace=0.1)
for f1 in range(n):
    for f2 in range(n):
        C = Nnorm_cost[:, :, jj, f2, f1]
        axs[f1, f2].imshow(C, cmap='hot', aspect='auto')
        axs[f1, f2].axis('off')
        plt.colorbar(axs[0, 0].imshow(Nnorm_cost[:, :, jj, f2, f1], cmap='hot', aspect='auto'))
plt.show()

# Visualize matrices of normalized cost
plt.figure()
fig2, axs = plt.subplots(n, n)
fig2.subplots_adjust(hspace=0.1, wspace=0.1)
for f1 in range(n):
    for f2 in range(n):
        C = All_cost[:, :, jj, f2, f1]
        axs[f1, f2].imshow(C, cmap='hot', aspect='auto')
        axs[f1, f2].axis('off')
        plt.colorbar(axs[0, 0].imshow(All_cost[:, :, jj, f2, f1], cmap='hot', aspect='auto'))
plt.show()

# Visualize Rij matrices
plt.figure()
fig3, axs = plt.subplots(n, n)
fig3.subplots_adjust(hspace=0, wspace=0)
for f1 in range(n):
    for f2 in range(n):
        C = R_line[:, :, jj, f2, f1]
        axs[f1, f2].imshow(C, cmap='hot', aspect='auto')
        axs[f1, f2].axis('off')
        plt.colorbar(axs[0, 0].imshow(R_line[:, :, jj, f2, f1], cmap='hot', aspect='auto'))
plt.show()

# im1 = 202;
# im2 = 203;
# z = [46.18, -450];  # perfect match

# im1 = 194;
# im2 = 197;
# z = [420, -6];     # perfect match 2
