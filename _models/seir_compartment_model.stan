functions {
  real[] seir(real t, real[] y, real[] theta, real[] x_r, int[] x_i) {
    real dydt[4];

    real beta = theta[x_i[1]];
    real a = theta[x_i[1] + 1];
    real gamma = theta[x_i[1] + 2];

    for (i in 1:x_i[1]) {
        if (t <= x_r[i]) {
          beta = theta[i];
          break;
        }
    }

    dydt[1] = -beta * y[1] * y[3];
    dydt[2] = beta * y[1] * y[3] - a * y[2];
    dydt[3] = a * y[2] - gamma * y[3];
    dydt[4] = gamma * y[3];

    return dydt;
  }
}

data {
  int<lower=1> n;
  int<lower=1> n_population;
  int<lower=0> n_knots;
  int y[n];
  real t0;
  real t[n];
  real knots[n_knots];
}

transformed data {
  real x_r[n_knots] = knots;
  int x_i[1] = {n_knots};
}

parameters {
  real<lower=0> theta[n_knots + 2];
  real<lower=0, upper=1> s0;
}

model {
  real lambda[n];
  real y_hat[n, 4];
  real y0[4];
  theta ~ gamma(3, 1);
  s0 ~ beta(0.5, 0.5);

  y0 = {s0, 0.0, 1.0 - s0, 0.0};
  y_hat = integrate_ode_rk45(seir, y0, t0, t, theta, x_r, x_i);

  for (i in 1:n)
    lambda[i] = y_hat[i, 3] * n_population;
  y ~ poisson(lambda);
}

generated quantities {
  real R0 = theta[1] / theta[3];
}
