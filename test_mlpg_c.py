import numpy as np
from mlpg_c import mlpg_c as mlpg

coeff = [-0.5,0.5,0.0]

st_mat = np.random.randn(200,5) # create a random "static" array with 200 length and 5 dimensions
dlt_mat = 0.1*np.random.randn(200,5) # create a random "delta" array with 200 length and 5 dimensions

jnt_sd_mat_1 = np.c_[st_mat, dlt_mat] # concatenate the random "static" and "delta" arrays
jnt_sd_mat_2 = np.concatenate([st_mat,np.insert(st_mat[:-1,:]*coeff[0], 0, 0.0, axis=0) + st_mat*coeff[1] + np.append(st_mat[1:,:]*coeff[2], np.zeros((1,st_mat.shape[1])), axis=0)], axis=1) # compute the "real delta" from the random "static", and concatenate

dcov_1 = np.var(jnt_sd_mat_1, axis=0) # variances of jnt_sd_mat_1
dcov_2 = np.var(jnt_sd_mat_2, axis=0) # variances of jnt_sd_mat_2

prec_1 = 1.0/dcov_1 # precisions of jnt_sd_mat_1
prec_2 = 1.0/dcov_2 # precisions of jnt_sd_mat_2

trj_gen_1 = mlpg.mlpg_solve(jnt_sd_mat_1, prec_1, np.array(coeff)) # trajectory considering static and delta values of jnt_sd_mat_1 --> should be smoother than that of the st_mat (similar to Kalman filtering)
trj_gen_2 = mlpg.mlpg_solve(jnt_sd_mat_2, prec_2, np.array(coeff)) # trajectory considering static and delta values of jnt_sd_mat_2 --> should be the same as that of the st_mat

# write txt
f = open('st_mat.txt', 'w')
for i in range(st_mat.shape[0]):
    for j in range(st_mat.shape[1]):
        if j < st_mat.shape[1]-1:
            f.write(str(st_mat[i][j])+' ')
        else:
            f.write(str(st_mat[i][j])+'\n')
f.close()

f = open('trj_gen_1.txt', 'w')
for i in range(trj_gen_1.shape[0]):
    for j in range(trj_gen_1.shape[1]):
        if j < trj_gen_1.shape[1]-1:
            f.write(str(trj_gen_1[i][j])+' ')
        else:
            f.write(str(trj_gen_1[i][j])+'\n')
f.close()

f = open('trj_gen_2.txt', 'w')
for i in range(trj_gen_2.shape[0]):
    for j in range(trj_gen_2.shape[1]):
        if j < trj_gen_2.shape[1]-1:
            f.write(str(trj_gen_2[i][j])+' ')
        else:
            f.write(str(trj_gen_2[i][j])+'\n')
f.close()
