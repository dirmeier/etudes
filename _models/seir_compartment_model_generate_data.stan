functions {
  real[] seir(real t, real[] y, real[] theta, real[] x_r, int[] x_i) {
    real dydt[4];
    real N = y[1] + y[2] + y[3] + y[4];

    real beta = theta[x_i[1]];
    real a = theta[x_i[1] + 1];
    real gamma = theta[x_i[1] + 2];

    for (i in 1:x_i[1]) {
        if (t <= x_r[i]) {
          beta = theta[i];
          break;
        }
    }

    dydt[1] = -beta * y[1] * y[3] / N;
    dydt[2] = beta * y[1] * y[3] / N - a * y[2];
    dydt[3] = a * y[2] - gamma * y[3];
    dydt[4] = gamma * y[3];

    return dydt;
  }
}

data {
  int<lower=0> n;
  real<lower=0> n_population;
  int<lower=0> n_knots;
  real t0;
  real t[n];
  real knots[n_knots];
  real theta[n_knots + 2];
}

transformed data {
  real x_r[n_knots] = knots;
  int x_i[1] = {n_knots};
  real y0[4] = {n_population - 1.0, 0.0, 1.0, 0.0};
}

generated quantities {
  real y_hat[n, 4];
  y_hat = integrate_ode_rk45(seir, y0, t0, t, theta, x_r, x_i);
}

