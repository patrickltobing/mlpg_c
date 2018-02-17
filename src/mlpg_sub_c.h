void c_calc_wum(const double* w, const double* u, const double* const * m, int T, int dim, double** wum);
void c_calc_wumseq(const double* w, const double* const * u, const double* const * m, int T, int dim, double** wum);
void c_calc_bandwuw(const double* w, const double* u, int T, int dim, int n_coeff, double** bandwuw);
void c_calc_bandwuwseq(const double* w, const double*  const * u, int T, int dim, int n_coeff, double** bandwuw);
void c_calc_cholband(const double* const * bandmat, int T, int dim, int n_diag, int length, double** Lmat);
void c_calc_cholsolve(const double* const * Lchol, const double* const * rhs, int T, int dim, int n_diag, double** y, double** x);
