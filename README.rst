mlpg_c
====

Maximum Likelihood Parameter Generation (MLPG) implementation in C for Python

Install
----

pip install mlpg_c

Usage
----

from mlpg_c import mlpg_c as mlpg

param_gen = mlpg.mlpg_solve(jnt_sd_mat, prec_vec, coef_vec)

Variable desc.
----

**param_gen**: generated parameter trajectory: T x dim

**jnt_sd_mat**: joint static-delta feature vector sequence: T x (dim*2)

**prec_vec**: vector of diagonal precision (inverse covariance) matrix: (dim*2) x 1

**coef_vec**: vector of delta coefficients: n_coeff x 1

To-do:
----

- time-varying diagonal precision matrices
- full precision matrix
- docs
