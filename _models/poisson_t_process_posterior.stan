data {
  int<lower=1> n;
  int<lower=1> n_obs;
    
  int<lower=1, upper=n> observed_idx[n_obs];
  int y_observed[n_obs];
  real x[n];

  real<lower=0> rho;
  real<lower=0> alpha;
  real<lower=0> nu;
}

transformed data {
  matrix[n, n] cov = cov_exp_quad(x, alpha, rho)
                     + diag_matrix(rep_vector(1e-10, n));
}

parameters {
  vector[n] tp;
}


model {
  tp ~ multi_student_t(nu, rep_vector(0, n), cov);  
  y_observed ~ poisson_log(tp[observed_idx]);
}

generated quantities {  
  vector[n] y;
  for (i in 1:n)
    y[i] = poisson_log_rng(tp[i]);
}
