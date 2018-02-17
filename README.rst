mlpg_c
====

Maximum Likelihood Parameter Generation (MLPG) implementation in C for Python

Install
----

pip install mlpg_c

Usage of mlpg_solve
----

from mlpg_c import mlpg_c as mlpg

param_gen = mlpg.mlpg_solve(jnt_sd_mat, prec_vec, coef_vec)

Variable desc.
----

**param_gen**: generated parameter trajectory: T x dim

**jnt_sd_mat**: joint static-delta feature vector sequence: T x (dim*2)

**prec_vec**: vector of diagonal precision (inverse covariance) matrix: (dim*2) x 1

**coef_vec**: vector of delta coefficients: n_coeff x 1. e.g.: [-0.5,0.5,0.0]

Usage of mlpg_solve_seq
----

from mlpg_c import mlpg_c as mlpg

param_gen = mlpg.mlpg_solve_seq(jnt_sd_mat, prec_mat, coef_vec)

Variable desc.
----

**param_gen**: generated parameter trajectory: T x dim

**jnt_sd_mat**: joint static-delta feature vector sequence: T x (dim*2)

**prec_mat**: sequence of diagonal precision (inverse covariance) matrices: T x (dim*2)

**coef_vec**: vector of delta coefficients: n_coeff x 1, e.g.: [-0.5,0.5,0.0]

To-do:
----

- demo
- full precision matrix
- docs
