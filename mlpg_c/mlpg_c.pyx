import cython
import numpy as np
cimport numpy as np

cdef extern from "mlpg_sub_c.h":
	void c_calc_wum(const double* w, const double* u, const double* const * m, int T, int dim, double** wum)
	void c_calc_wumseq(const double* w, const double* const * u, const double* const * m, int T, int dim, double** wum)
	void c_calc_bandwuw(const double* w, const double* u, int T, int dim, int n_coeff, double** bandwuw)
	void c_calc_bandwuwseq(const double* w, const double*  const * u, int T, int dim, int n_coeff, double** bandwuw)
	void c_calc_cholband(const double* const * bandmat, int T, int dim, int n_diag, int length, double** Lmat)
	void c_calc_cholsolve(const double* const * Lchol, const double* const * rhs, int T, int dim, int n_diag, double** y, double** x)

def calc_wum(np.ndarray[double, ndim=1, mode="c"] coeff not None, np.ndarray[double, ndim=1, mode="c"] prec not None, np.ndarray[double, ndim=2, mode="c"] sd_mat not None):
	cdef int T, dim, n_diag

	T, dim, n_diag = sd_mat.shape[0], int(len(prec)/2), len(coeff)

	cdef double[::1] coeff_data = coeff
	cdef double[::1] prec_data = prec
	cdef double[:, ::1] sdmat_data = sd_mat
	cdef double[:, ::1] wum_data = np.zeros((T, dim), dtype=np.dtype('float64'))

	cdef np.intp_t[:] tmp1 = np.zeros(n_diag, dtype=np.intp)
	cdef np.intp_t[:] tmp2 = np.zeros(dim, dtype=np.intp)
	cdef np.intp_t[:] tmp3 = np.zeros(T, dtype=np.intp)
	cdef np.intp_t[:] tmp4 = np.zeros(T, dtype=np.intp)

	cdef double* cpp_coeff = <double*> (<void*> &tmp1[0])
	cdef double* cpp_prec = <double*> (<void*> &tmp2[0])
	cdef double** cpp_sdmat = <double**> (<void*> &tmp3[0])
	cdef double** cpp_wum = <double**> (<void*> &tmp4[0])

	cpp_coeff = &coeff_data[0]
	cpp_prec = &prec_data[0]

	cdef np.intp_t i

	for i in range(T):
		cpp_sdmat[i] = &sdmat_data[i, 0]
		cpp_wum[i] = &wum_data[i, 0]

	c_calc_wum(cpp_coeff, cpp_prec, cpp_sdmat, T, dim, cpp_wum)

	return np.array(wum_data, dtype=np.float64)

def calc_wumseq(np.ndarray[double, ndim=1, mode="c"] coeff not None, np.ndarray[double, ndim=2, mode="c"] prec not None, np.ndarray[double, ndim=2, mode="c"] sd_mat not None):
	cdef int T, dim, n_diag

	T, dim, n_diag = sd_mat.shape[0], int(len(prec)/2), len(coeff)

	cdef double[::1] coeff_data = coeff
	cdef double[:, ::1] prec_data = prec
	cdef double[:, ::1] sdmat_data = sd_mat
	cdef double[:, ::1] wum_data = np.zeros((T, dim), dtype=np.dtype('float64'))

	cdef np.intp_t[:] tmp1 = np.zeros(n_diag, dtype=np.intp)
	cdef np.intp_t[:] tmp2 = np.zeros(T, dtype=np.intp)
	cdef np.intp_t[:] tmp3 = np.zeros(T, dtype=np.intp)
	cdef np.intp_t[:] tmp4 = np.zeros(T, dtype=np.intp)

	cdef double* cpp_coeff = <double*> (<void*> &tmp1[0])
	cdef double** cpp_prec = <double**> (<void*> &tmp2[0])
	cdef double** cpp_sdmat = <double**> (<void*> &tmp3[0])
	cdef double** cpp_wum = <double**> (<void*> &tmp4[0])

	cpp_coeff = &coeff_data[0]

	cdef np.intp_t i

	for i in range(T):
		cpp_prec[i] = &prec_data[i, 0]
		cpp_sdmat[i] = &sdmat_data[i, 0]
		cpp_wum[i] = &wum_data[i, 0]

	c_calc_wumseq(cpp_coeff, cpp_prec, cpp_sdmat, T, dim, cpp_wum)

	return np.array(wum_data, dtype=np.float64)

def calc_bandwuw(np.ndarray[double, ndim=1, mode="c"] coeff not None, np.ndarray[double, ndim=1, mode="c"] prec not None, int T):
	cdef int dim, n_diag

	dim, n_diag = int(len(prec)/2), len(coeff)

	cdef double[::1] coeff_data = coeff
	cdef double[::1] prec_data = prec
	cdef double[:, ::1] bandwuw_data = np.zeros((n_diag, T*dim), dtype=np.dtype('float64'))

	cdef np.intp_t[:] tmp1 = np.zeros(n_diag, dtype=np.intp)
	cdef np.intp_t[:] tmp2 = np.zeros(dim, dtype=np.intp)
	cdef np.intp_t[:] tmp3 = np.zeros(n_diag, dtype=np.intp)

	cdef double* cpp_coeff = <double*> (<void*> &tmp1[0])
	cdef double* cpp_prec = <double*> (<void*> &tmp2[0])
	cdef double** cpp_bandwuw = <double**> (<void*> &tmp3[0])

	cpp_coeff = &coeff_data[0]
	cpp_prec = &prec_data[0]

	cdef np.intp_t i

	for i in range(n_diag):
		cpp_bandwuw[i] = &bandwuw_data[i, 0]

	c_calc_bandwuw(cpp_coeff, cpp_prec, T, dim, n_diag, cpp_bandwuw)

	return np.array(bandwuw_data, dtype=np.float64)

def calc_bandwuwseq(np.ndarray[double, ndim=1, mode="c"] coeff not None, np.ndarray[double, ndim=2, mode="c"] prec not None, int T):
	cdef int dim, n_diag

	dim, n_diag = int(len(prec)/2), len(coeff)

	cdef double[::1] coeff_data = coeff
	cdef double[:, ::1] prec_data = prec
	cdef double[:, ::1] bandwuw_data = np.zeros((n_diag, T*dim), dtype=np.dtype('float64'))

	cdef np.intp_t[:] tmp1 = np.zeros(n_diag, dtype=np.intp)
	cdef np.intp_t[:] tmp2 = np.zeros(dim, dtype=np.intp)
	cdef np.intp_t[:] tmp3 = np.zeros(n_diag, dtype=np.intp)

	cdef double* cpp_coeff = <double*> (<void*> &tmp1[0])
	cdef double** cpp_prec = <double**> (<void*> &tmp2[0])
	cdef double** cpp_bandwuw = <double**> (<void*> &tmp3[0])

	cpp_coeff = &coeff_data[0]

	cdef np.intp_t i

	for i in range(T):
		cpp_prec[i] = &prec_data[i, 0]

	for i in range(n_diag):
		cpp_bandwuw[i] = &bandwuw_data[i, 0]

	c_calc_bandwuwseq(cpp_coeff, cpp_prec, T, dim, n_diag, cpp_bandwuw)

	return np.array(bandwuw_data, dtype=np.float64)

def calc_cholband(np.ndarray[double, ndim=2, mode="c"] bandwuw not None, int T):
	cdef int dim, n_diag, length

	dim, n_diag, length = int(bandwuw.shape[1]/T), bandwuw.shape[0], bandwuw.shape[1]

	cdef double[:, ::1] bandwuw_data = bandwuw
	cdef double[:, ::1] Lmat_data = np.zeros((n_diag, length), dtype=np.dtype('float64'))

	cdef np.intp_t[:] tmp1 = np.zeros(n_diag, dtype=np.intp)
	cdef np.intp_t[:] tmp2 = np.zeros(n_diag, dtype=np.intp)

	cdef double** cpp_bandwuw = <double**> (<void*> &tmp1[0])
	cdef double** cpp_Lmat = <double**> (<void*> &tmp2[0])

	cdef np.intp_t i

	for i in range(n_diag):
		cpp_bandwuw[i] = &bandwuw_data[i, 0]
		cpp_Lmat[i] = &Lmat_data[i, 0]

	c_calc_cholband(cpp_bandwuw, T, dim, n_diag, length, cpp_Lmat)

	return np.array(Lmat_data, dtype=np.float64)

def calc_cholsolve(np.ndarray[double, ndim=2, mode="c"] cholbandwuw not None, np.ndarray[double, ndim=2, mode="c"] wum not None):
	cdef int T, dim, n_diag

	T, dim, n_diag = wum.shape[0], wum.shape[1], cholbandwuw.shape[0]

	cdef double[:, ::1] cholbandwuw_data = cholbandwuw
	cdef double[:, ::1] wum_data = wum
	cdef double[:, ::1] y_data = np.zeros((T, dim), dtype=np.dtype('float64'))
	cdef double[:, ::1] x_data = np.zeros((T, dim), dtype=np.dtype('float64'))

	cdef np.intp_t[:] tmp1 = np.zeros(n_diag, dtype=np.intp)
	cdef np.intp_t[:] tmp2 = np.zeros(T, dtype=np.intp)
	cdef np.intp_t[:] tmp3 = np.zeros(T, dtype=np.intp)
	cdef np.intp_t[:] tmp4 = np.zeros(T, dtype=np.intp)

	cdef double** cpp_cholbandwuw = <double**> (<void*> &tmp1[0])
	cdef double** cpp_wum = <double**> (<void*> &tmp2[0])
	cdef double** cpp_y = <double**> (<void*> &tmp3[0])
	cdef double** cpp_x = <double**> (<void*> &tmp4[0])

	cdef np.intp_t i

	for i in range(n_diag):
		cpp_cholbandwuw[i] = &cholbandwuw_data[i, 0]

	for i in range(T):
		cpp_wum[i] = &wum_data[i, 0]
		cpp_y[i] = &y_data[i, 0]
		cpp_x[i] = &x_data[i, 0]

	c_calc_cholsolve(cpp_cholbandwuw, cpp_wum, T, dim, n_diag, cpp_y, cpp_x)

	return np.array(x_data, dtype=np.float64)

def mlpg_solve(np.ndarray[double, ndim=2, mode="c"] sd_mat not None, np.ndarray[double, ndim=1, mode="c"] prec not None, np.ndarray[double, ndim=1, mode="c"] coeff not None):
	assert sd_mat.ndim == 2, "sd_mat.ndim = %d != 2" % sd_mat.ndim
	assert sd_mat.shape[1]%2 == 0, "sd_mat.shape[1]%%2 = %d != 0" % (sd_mat.shape[1]%2)
	assert prec.ndim == 1, "prec.ndim = %d != 1" % prec.ndim
	assert sd_mat.shape[1] == len(prec), "sd_mat.shape[1] = %d != %d = len(prec)" % (sd_mat.shape[1], len(prec))
	assert coeff.ndim == 1, "coeff.ndim = %d != 1" % coeff.ndim
	assert len(coeff) == 3, "len(coeff) = %d != 3" % len(coeff)

	cdef int T = sd_mat.shape[0]
	
	sd_mat = np.array(sd_mat, dtype=np.float64)
	prec = np.array(prec, dtype=np.float64)
	coeff = np.array(coeff, dtype=np.float64)

	wum = calc_wum(coeff, prec, sd_mat)
	bandwuw = calc_bandwuw(coeff, prec, T)
	cholbandwuw = calc_cholband(bandwuw, T)
	x = calc_cholsolve(cholbandwuw, wum)

	return x

def mlpg_solve_seq(np.ndarray[double, ndim=2, mode="c"] sd_mat not None, np.ndarray[double, ndim=2, mode="c"] prec not None, np.ndarray[double, ndim=1, mode="c"] coeff not None):
	assert sd_mat.ndim == 2, "sd_mat.ndim = %d != 2" % sd_mat.ndim
	assert sd_mat.shape[1]%2 == 0, "sd_mat.shape[1]%%2 = %d != 0" % (sd_mat.shape[1]%2)
	assert prec.ndim == 2, "prec.ndim = %d != 2" % prec.ndim
	assert sd_mat.shape[0] == prec.shape[0], "sd_mat.shape[0] = %d != %d = prec.shape[0]" % (sd_mat.shape[0], prec.shape[0])
	assert sd_mat.shape[1] == prec.shape[1], "sd_mat.shape[1] = %d != %d = prec.shape[1]" % (sd_mat.shape[1], prec.shape[1])
	assert coeff.ndim == 1, "coeff.ndim = %d != 1" % coeff.ndim
	assert len(coeff) == 3, "len(coeff) = %d != 3" % len(coeff)

	cdef int T = sd_mat.shape[0]
	
	sd_mat = np.array(sd_mat, dtype=np.float64)
	prec = np.array(prec, dtype=np.float64)
	coeff = np.array(coeff, dtype=np.float64)

	wum = calc_wumseq(coeff, prec, sd_mat)
	bandwuw = calc_bandwuwseq(coeff, prec, T)
	cholbandwuw = calc_cholband(bandwuw, T)
	x = calc_cholsolve(cholbandwuw, wum)

	return x
