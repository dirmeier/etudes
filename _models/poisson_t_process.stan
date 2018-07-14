data {
  int<lower=1> n;
  real x[n];
  real<lower=0> rho;
  real<lower=0> alpha;
  real<lower=0> nu;
}

transformed data {
  matrix[n, n] cov = cov_exp_quad(x, alpha, rho)
                     + diag_matrix(rep_vector(1e-10, n));
}

parameters {}
model {}

generated quantities {
  vector[n] tp = multi_student_t_rng(nu, rep_vector(0, n), cov);  
  vector[n] y;
  for (i in 1:n) {    
    y[i] = poisson_log_rng(tp[i]);    
  }
}
