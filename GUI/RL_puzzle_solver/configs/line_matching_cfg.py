### LINE MATCHING
thr_coef = 0.08
max_dist = 30 #3                        xy_step                 ~ xy_step
badmatch_penalty = 50 #30               max_dist * 5/3          ~ fragment size would be ideal
mismatch_penalty = 40 #20 #10           max_dist * 4/3          ~ 1.5 time xy_step
rmax = 35 #15 #11                       max_dist * 10 / 3       ~ almost_max_dist but slightly higher


#max_dist=30, badmatch_penalty=50 mismatch_penalty=40 rmax=35
