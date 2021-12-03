#include functions.stan

data {
  int<lower=0> N_I;
  int<lower=0> N_O;
  int<lower=0, upper=N_O> i_to_o[N_I];

  vector[N_I] X;
  vector[N_I] tr;
  vector[N_I] y;

  real<lower=0> alpha;
  real<lower=0> beta;
  int<lower=0> degree;
  int<lower=0> n_knots;

  vector[n_knots] u_knots;
  vector[n_knots] x_knots;
  vector[n_knots] tr_knots;
}

transformed data {
  int n_coefs = n_knots - degree - 1;

  real xr[N_I]  = to_array_1d(X);
  real trr[N_I] = to_array_1d(tr);

  vector[n_knots] knots[3];
  knots[1] = u_knots;
  knots[2] = x_knots;
  knots[3] = tr_knots;
}


parameters {
  real<lower=0> u_scale;
  vector[N_O]   U_tilde;

  real x_beta;
  real<lower=0> x_scale;

  real<lower=0> t_beta_tau;
  vector[n_coefs * n_coefs] t_beta;
  real<lower=0> t_scale;

  real<lower=0> y_beta_tau;
  vector[n_coefs * n_coefs * n_coefs] y_beta;
  real<lower=0>  y_scale;
}


transformed parameters {
  vector[N_O] U = U_tilde * u_scale;
}

model {
  real ur[N_I] = to_array_1d(U[i_to_o]);
  vector[2] uxr[N_I]  = concat_rr(N_I, ur, xr);
  vector[3] uxtr[N_I] = concat_vr(N_I, uxr, trr);

  matrix[N_I, n_coefs * n_coefs] t_design  = tensor_spline(
    uxr,  n_coefs, degree, knots
  );
  matrix[N_I, n_coefs * n_coefs * n_coefs] y_design  = tensor_spline(
    uxtr, n_coefs, degree, knots
  );

  u_scale ~ std_normal();
  U_tilde ~ std_normal();

  x_beta ~ std_normal();
  x_scale ~ std_normal();

  t_beta_tau ~ std_normal();
  t_beta[1] ~ std_normal();
  for (i in 2:size(t_beta)) {
     t_beta[i] ~ normal(t_beta[i - 1], t_beta_tau);
  }
  t_scale ~ std_normal();

  y_beta_tau ~ std_normal();
  y_beta[1] ~ std_normal();
  for (i in 2:size(y_beta)) {
     y_beta[i] ~ normal(y_beta[i - 1], y_beta_tau);
  }
  y_scale ~ std_normal();

  xr  ~ normal(U[i_to_o], x_scale);
  tr  ~ normal(t_design * t_beta, t_scale);
  y   ~ normal(y_design * y_beta, y_scale);
}

generated quantities {
  vector[N_I] ite;
  {
      real      tr1[N_I]   = to_array_1d(tr + 1);
      real      ur[N_I]    = to_array_1d(U[i_to_o]);
      vector[2] uxr[N_I]   = concat_rr(N_I, ur, xr);
      vector[3] uxtr1[N_I] = concat_vr(N_I, uxr, tr1);

      matrix[N_I, n_coefs * n_coefs * n_coefs] y_star_design = tensor_spline(
        uxtr1, n_coefs, degree, knots
      );

      vector[N_I] y_star_mu  = y_star_design * y_beta;

      for (i in 1:N_I)
          ite[i] = normal_rng(y_star_mu[i] - y[i], y_scale);
  }
}


