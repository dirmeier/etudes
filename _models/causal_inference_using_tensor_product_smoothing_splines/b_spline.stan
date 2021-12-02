functions {
  real bspline_basis(real x, vector knots, int i, int d)
  {
    int n_knots = size(knots);
    real b1 = 0;
    real b2 = 0;

    if (d == 0)
      return knots[i] <= x && x < knots[i + 1] ? 1.0 : 0.0;

    if (knots[i + d] != knots[i]) {
      b1 = (x - knots[i]) / (knots[i + d] - knots[i]);
      b1 = b1 * bspline_basis(x, knots, i, d - 1);
    }

    if (knots[i + d + 1] != knots[i + 1]) {
      b2 = (knots[i + d + 1] - x) / (knots[i + d + 1] - knots[i + 1]);
      b2 = b2 * bspline_basis(x, knots, i + 1, d - 1);
    }

    return b1 + b2;
  }

  matrix bspline(vector u, int n_coef, int d, vector knots)
  {
      int n = size(u);
      matrix[n, n_coef] mu;

      if (n_coef + d + 1 != size(knots))
        reject("n_coef + d != size(knots)");

      for (i in 1:n) {
          for (j in 1:n_coef) {
              mu[i, j] = bspline_basis(u[i], knots, j, d);
          }
      }

      return mu;
  }
}

data {
  int<lower=0> N;
  vector[N] X;
  vector[N] y;

  int<lower=0> degree;
  int<lower=0> n_knots;
  vector[n_knots] knots;
}

transformed data {
  int n_coef = n_knots - degree - 1;
  matrix[N, n_coef] B = bspline(X, n_coef, degree, knots);
}


parameters {
  vector[n_coef] mu;
  real<lower=0> mu_tau;
  real<lower=0> sigma;
}

model {
  mu_tau ~ inv_gamma(4, 4);
  mu[1] ~ std_normal();
  for (i in 2:n_coef) {
      mu[i] ~ normal(mu[i - 1], mu_tau);
  }

  y ~ normal(B * mu, sigma);
}

generated quantities {
  vector[N] y_hat;
  y_hat = B * mu;
}


