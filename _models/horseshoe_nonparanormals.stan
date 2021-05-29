functions {
  matrix matrix_from_vec(vector nd, vector d, int K) {
    matrix[K, K] y;
    int pos = 1;

    for (i in 1:K) { y[i, i] = d[i]; }
    for (i in 1:(K - 1)) {
      for (j  in (i + 1):K) {
        y[i, j] = nd[pos];
        y[j, i] = nd[pos];
        pos += 1;
      }
    }
    return y;
  }
}


data {
  int<lower=1> N;
  int<lower=1> P;
  int<lower=1> P_upper;
  vector[P] y[N];
}

transformed data {
  real slab_scale = 3;
  real slab_scale_square = square(slab_scale);
  real slab_df = 25;
  real half_slab_df = 0.5 * slab_df;
}

parameters {
  vector<lower=0>[P] omega_diag;

  vector[P_upper] omega_tilde;
  vector<lower=0>[P_upper] lambda;
  real<lower=0> c2_tilde;
  real<lower=0> tau;
}

transformed parameters {
  matrix[P, P] Omega;
  vector[P_upper] omega;
  {
    real c2 = slab_scale_square * c2_tilde;

    vector[P_upper] lambda_tilde =
      sqrt( c2 * square(lambda) ./ (c2 + square(tau) * square(lambda)) );
    omega = omega_tilde .* lambda_tilde * tau;

    Omega = matrix_from_vec(omega, omega_diag, P) +
      diag_matrix(rep_vector(1e-6, P));
  }
}

model {
  omega_diag ~ inv_gamma(1, 1);
  omega_tilde ~ normal(0, 1);
  lambda ~ cauchy(0, 1);
  tau ~ cauchy(0, 1);
  c2_tilde ~ inv_gamma(half_slab_df, half_slab_df);

  y ~ multi_normal_prec(rep_vector(0, P), Omega);
}
