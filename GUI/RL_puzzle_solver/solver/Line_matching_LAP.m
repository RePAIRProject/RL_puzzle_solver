import matplotlib.pyplot as plt
import numpy as np

def translation2(beta, R, p):
    ## a*x + b*y + c = 0  - initial equation of the line
    a = np.cos(beta)
    b = np.sin(beta)
    c = -R

    if b == 0:  ## if beta 90°
        y = np.linspace(-2000, 5000)
        x = -c/a * np.ones_like(y)
    else:
        x = np.linspace(-2000, 5000)
        y = 1/b * (-a*x - c)

    ## a*(x+p(1)) + b*(y+p(2)) + c = 0 -  equation of translated line
    c_new = (a*p[0] + b*p[1] + c)
    if b == 0:
        y_new = np.linspace(-2000, 5000)
        x = -c_new/a * np.ones_like(y)
    else:
        y_new = 1/b * (-a*x - c_new)
    R_new = -c_new
    ## if or (and((1/b*(-a*p(1)-c)>p(2)), (1/b*(-a*0-c)>0)), and((1/b*(-a*p(1)-c)<p(2)), (1/b*(-a*0-c)<0)))
    if ((1/b*(-a*p[0]-c) > p[1]) and (1/b*(-a*0-c) > 0)) or ((1/b*(-a*p[0]-c) < p[1]) and (1/b*(-a*0-c) < 0)):
        beta_new = beta
    else:
        R_new = -R_new
        beta_new = (beta + np.pi) % (2*np.pi)
    return beta_new, R_new, x, y, y_new

## MAIN
rmax = 50
tr_coef = 0.1
max_dist = 1000
ang = 45
step = 0.04
m = []

z_rad = 1000     #size of grid
p = [500, 500]   #center of grid (0,0)
m[:,:,1] = np.meshgrid(np.arange(-1, 1+step, step)).T #m(:,:,1) = meshgrid(-1:step:1)
m[:,:,2] = np.meshgrid(np.arange(-1, 1+step, step))  #??? wrong???

z_id = m*z_rad
grid_step = z_id[1,0,0]-z_id[0,0,0]
rot = np.arange(0, 360-ang+1, ang)

pieces = np.concatenate((np.arange(194, 199), np.arange(200, 204)))
All_cost = np.zeros((m.shape[1], m.shape[1], len(rot), len(pieces), len(pieces)))

for f1 in range(1, 10):          # - select fixed fragment
    # read image
    im1 = pieces[f1-1]
    alfa1, R1, I1 = read_info(im1)
    for f2 in range(1, 10):      # - select moving and rotating fragment
        # read image
        im2 = pieces[f2-1]
        alfa2, R2, I2 = read_info(im2)
        R_cost = np.zeros((m.shape[1], m.shape[1], rot.shape[1]))
        if f1 == f2:
            R_norm = R_cost - 1
            R_norm2 = R_norm
        else:
            for t in range(len(rot)):
                teta = -rot[t] * np.pi / 180  # rotation of F2
                for ix in range(z_id.shape[0]):
                    for iy in range(z_id.shape[0]):
                        z = z_id[ix, iy]
                        n_linesf1 = alfa1.shape[0]
                        n_linesf2 = alfa2.shape[0]
                        cost_matrix = np.zeros((n_linesf1, n_linesf2))

                        for i in range(n_linesf1):
                            for j in range(n_linesf2):
                                # translate reference point to the center
                                beta1, R_new1, x1, y1, y_new1 = translation2(alfa1[i], R1[i], p)
                                beta2, R_new2, x2, y2, y_new2 = translation2(alfa2[j], R2[j], p)
                                beta3, R_new3, x3, y3, y_new3 = translation2(beta2 + teta, R_new2, -z)  ## shift and rot line 2

                                dist1 = np.sqrt((R_new1**2 + R_new3**2 - 2*np.abs(R_new1*R_new3)*np.cos(gamma)))
                                dist2 = np.sqrt((R_new2**2 + R_new4**2 - 2*np.abs(R_new2*R_new4)*np.cos(gamma)))
                                if coef < tr_coef:
                                    cost = (dist1 + dist2)
                                else:
                                    cost = max_dist
                                cost_matrix[i, j] = cost
                                #end
                            #end
                        #end
                        assignment, tot_cost = munkres(cost_matrix)  ##name of function???
                        R_cost[iy, ix, t] = tot_cost
                        ## option
                        r_c = np.min(np.sum(assignment*cost_matrix))
                        R_cost2[iy, ix, t] = r_c
                    #end
                #end
            #end
            R_norm = np.maximum(1 - R_cost/rmax, 0)
            R_norm2 = np.maximum(1 - R_cost2/rmax, 0)
        #end
        Nnorm_cost[:, :, :, f2, f1] = R_cost
        All_cost[:, :, :, f2, f1] = R_norm
        All_cost2[:, :, :, f2, f1] = R_norm2
    #end
#end

R_mask = np.load('R_mask51_45cont2.mat')
rneg = np.where(R_mask<0, -1, 0)

R_line = All_cost*R_mask+rneg
R_line2 = All_cost2*R_mask+rneg
for jj in range(9):
    R_line[:,:,:,jj,jj] = -1

R_line = R_line*2
R_line[R_line<0] = -0.5
for jj in range(9):
    R_line2[:,:,:,jj,jj] = -1
R_line2 = R_line2*2
R_line2[R_line2<0] = -0.5
np.save('R_line51_45_verLAP_fake2.npy', R_line2)
np.save('R_line51_45_verLAP_fakeNEW.npy', R_line) # full LAP

#########################################################
## Visualize Rij matrices

## Visualize matrices of cost before norm
fig, axs = plt.subplots(9, 9)
fig.subplots_adjust(hspace=0, wspace=0)
for f1 in range(9):
    for f2 in range(9):
        axs[f1, f2].imshow(Nnorm_cost[:, :, jj, f2, f1], cmap='hot', aspect='auto')
        axs[f1, f2].axis('off')
plt.colorbar(axs[0, 0].imshow(Nnorm_cost[:, :, jj, f2, f1], cmap='hot', aspect='auto'))
plt.show()

## Visualize matrices of normalized cost
fig, axs = plt.subplots(9, 9)
fig.subplots_adjust(hspace=0, wspace=0)
for f1 in range(9):
    for f2 in range(9):
        axs[f1, f2].imshow(All_cost[:, :, jj, f2, f1], cmap='hot', aspect='auto')
        axs[f1, f2].axis('off')
plt.colorbar(axs[0, 0].imshow(All_cost[:, :, jj, f2, f1], cmap='hot', aspect='auto'))
plt.show()

## Visualize Rij matrices
fig, axs = plt.subplots(9, 9)
fig.subplots_adjust(hspace=0, wspace=0)
for f1 in range(9):
    for f2 in range(9):
        axs[f1, f2].imshow(R_line[:, :, jj, f2, f1], cmap='hot', aspect='auto')
        axs[f1, f2].axis('off')
plt.colorbar(axs[0, 0].imshow(R_line[:, :, jj, f2, f1], cmap='hot', aspect='auto'))
plt.show()



# im1 = 202;
# im2 = 203;
# z = [46.18, -450];  %% perfect match

# im1 = 194;
# im2 = 197;
# z = [420, -6];     %% perfect match 2
