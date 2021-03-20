data {
  int<lower=1> n;
  real x[n];
  vector[n] y;
  int<lower=1> n_star;
  real x_star[n_star];

  real<lower=0> alpha;
  real<lower=0> rho;
  real<lower=0> sigma;
}

parameters {}
model {}

generated quantities {
  vector[n_star] f_star;
  {
    matrix[n, n] K =  cov_exp_quad(x, alpha, rho)
        + diag_matrix(rep_vector(1e-10, n));
    matrix[n_star, n] K_star =  cov_exp_quad(x_star, x, alpha, rho);
    matrix[n_star, n_star] K_star_star =  cov_exp_quad(x_star, alpha, rho)
        + diag_matrix(rep_vector(1e-10, n_star));

    matrix[n, n] K_sigma = K
        + diag_matrix(rep_vector(square(sigma), n));
    matrix[n, n] K_sigma_inv = inverse(K_sigma);

    f_star = multi_normal_rng(
      K_star * K_sigma_inv * y,
      K_star_star - (K_star * K_sigma_inv * K_star')
    );
  }
}
