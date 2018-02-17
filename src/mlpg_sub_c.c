#include "mlpg_sub_c.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void c_calc_wum(const double* w, const double* u, const double* const * m, int T, int dim, double** wum) {
	int i, j, jdiff;

	for (i = 0; i < T; i++)
		for (j = 0; j < dim; j++) {
			jdiff = j + dim;
			wum[i][j] = u[j] * m[i][j] + w[1] * u[jdiff] * m[i][jdiff];
			if ((i+1) < T)
				wum[i][j] += w[0] * u[jdiff] * m[i+1][jdiff];
			if ((i-1) >= 0)
				wum[i][j] += w[2] * u[jdiff] * m[i-1][jdiff];
		}

	return;
}

void c_calc_wumseq(const double* w, const double* const * u, const double* const * m, int T, int dim, double** wum) {
	int i, j, jdiff;

	for (i = 0; i < T; i++)
		for (j = 0; j < dim; j++) {
			jdiff = j + dim;
			wum[i][j] = u[i][j] * m[i][j] + w[1] * u[i][jdiff] * m[i][jdiff];
			if ((i+1) < T)
				wum[i][j] += w[0] * u[i+1][jdiff] * m[i+1][jdiff];
			if ((i-1) >= 0)
				wum[i][j] += w[2] * u[i-1][jdiff] * m[i-1][jdiff];
		}

	return;
}

void c_calc_bandwuw(const double* w, const double* u, int T, int dim, int n_coeff, double** bandwuw) {
	int i, j, k, jdiff, idim, jidim, n_diag = n_coeff;

	for (i = 0; i < T; i++)
		for (j = 0, idim = i*dim; j < dim; j++)
			for (k = 0, jidim = j+idim, bandwuw[0][jidim]=u[j]; k < n_diag; k++) {
				jdiff = j + dim;
				if (k == 0) {
					bandwuw[k][jidim] += w[1] * u[jdiff] * w[1];
					if ((i+1) < T)
						bandwuw[k][jidim] += w[0] * u[jdiff] * w[0];
					if ((i-1) >= 0)
						bandwuw[k][jidim] += w[2] * u[jdiff] * w[2];
				} else if ((i+k) < T) {
					if (k == 1)
						bandwuw[k][jidim] = u[jdiff] * (w[0] * w[1] + w[1] * w[2]);
					else if (k == 2)
						bandwuw[k][jidim] = w[0] * u[jdiff] * w[2];
				}
			}

	return;
}

void c_calc_bandwuwseq(const double* w, const double*  const * u, int T, int dim, int n_coeff, double** bandwuw) {
	int i, j, k, jdiff, idim, jidim, n_diag = n_coeff;

	for (i = 0; i < T; i++)
		for (j = 0, idim = i*dim; j < dim; j++)
			for (k = 0, jidim = j+idim, bandwuw[0][jidim]=u[i][j]; k < n_diag; k++) {
				jdiff = j + dim;
				if (k == 0) {
					bandwuw[k][jidim] += w[1] * u[i][jdiff] * w[1];
					if ((i+1) < T)
						bandwuw[k][jidim] += w[0] * u[i+1][jdiff] * w[0];
					if ((i-1) >= 0)
						bandwuw[k][jidim] += w[2] * u[i-1][jdiff] * w[2];
				} else if ((i+k) < T) {
					if (k == 1)
						bandwuw[k][jidim] = w[0] * u[i+1][jdiff] * w[1] + w[1] * u[i][jdiff] * w[2];
					else if (k == 2)
						bandwuw[k][jidim] = w[0] * u[i+1][jdiff] * w[2];
				}
			}

	return;
}

void c_calc_cholband(const double* const * bandmat, int T, int dim, int n_diag, int length, double** Lmat) {
	int i, j, k, l, d, t, dtdim, didim, dkdim;
	double sub, elmt;

	for (d = 0; d < dim; d++)
		for (t = 0; t < T; t++)
			for (i = 0, dtdim = t*dim+d; i < n_diag; i++) {
				didim = (t+i)*dim+d;
				if (i == 0) {
					for (j = 1, sub = 0.0; j < n_diag; j++) {
						k = t-j;
						if (k >= 0)
							sub += pow(Lmat[j][k*dim+d],2);
					}
					elmt = bandmat[i][dtdim]-sub;
					if (elmt <= 0) {
						fprintf(stderr, "error c_calc_cholband: cholesky diag[%d][%d]=%lf <= 0\n", t, d, elmt);
						exit(1);
					}
					Lmat[i][dtdim] = sqrt(elmt);
				} else if (didim < length) {
					for (j = (i+1), sub = 0.0, l = 1; j < n_diag; j++, l++) {
						k = t-l;
						if (k >= 0) {
							dkdim = k*dim+d;
							sub += Lmat[j][dkdim]*Lmat[l][dkdim];
						}
					}
					Lmat[i][dtdim] = (bandmat[i][dtdim]-sub)/Lmat[0][dtdim];
				}
			}

	return;
}

void c_calc_cholsolve(const double* const * Lchol, const double* const * rhs, int T, int dim, int n_diag, double** y, double** x) {
	int i, j, k, t, jidim;
	double sub;

	for (j = 0; j < dim; j++) {
		for (i = 0; i < T; i++) {
			for (k = 1, sub = 0.0; k < n_diag; k++) {
				t = i-k;
				if (t >= 0)
					sub += Lchol[k][t*dim+j] * y[t][j];
			}
			y[i][j] = (rhs[i][j]-sub)/Lchol[0][i*dim+j];
		}
		for (i = (T-1); i > -1; i--) {
			for (k = 1, jidim = j+i*dim, sub = 0.0; k < n_diag; k++) {
				t = i+k;
				if (t < T)
					sub += Lchol[k][jidim] * x[t][j];
			}
			x[i][j] = (y[i][j]-sub)/Lchol[0][jidim];
		}
	}

	return;
}
