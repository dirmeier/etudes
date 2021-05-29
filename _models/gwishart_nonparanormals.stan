functions {
  matrix matrix_from_vec(vector nd, vector d, int K) {
    matrix[K, K] y;
    int pos = 1;

    for (i in 1:K) { y[i, i] = 1; }
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
  vector[P] y[N];
}

transformed data {
  int Pl = (P * (P - 1) / 2);
}

parameters {
  vector[P] mu;

  vector<lower=0>[P] omega_diag;
  vector[Pl] omega_tilde;
  vector<lower=0>[Pl] lambda;
  real<lower=0> tau;
}

transformed parameters {
  cov_matrix[P] Omega;
  vector[Pl] omega = omega_tilde .* lambda * tau;
  {
    Omega = matrix_from_vec(omega, omega_diag, P) +
      diag_matrix(rep_vector(1e-6, P));
  }
}

model {
  lambda ~ cauchy(0, 1);
  omega_diag ~ cauchy(0, 1);
  omega_tilde ~ normal(0, 1);
  tau ~ cauchy(0, 1);

  y ~ multi_normal_prec(mu, Omega);
}
