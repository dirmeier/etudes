functions {


 vector reparam(vector f_tilde, real[] x, real alpha, real rho, real delta)  {    
    int N = num_elements(x);
    vector[N] f;
    matrix[N, N] L_K;
    matrix[N, N] K = cov_exp_quad(x, alpha, rho);

    // diagonal elements
    for (n in 1:N)
      K[n, n] = K[n, n] + delta;

    L_K = cholesky_decompose(K);
    f = L_K * f_tilde;
    return f;
 }

  vector pred_binary(vector f_tilde, real[] x, real[] x_new, real alpha, real rho, real delta) {
    int N = size(x);
    int N_new = size(x_new);
    matrix[N, N] K = cov_exp_quad(x, alpha, rho);
    matrix[N, N_new] K_new = cov_exp_quad(x, x_new, alpha, rho);
    vector[N] f_post = reparam(f_tilde, x, alpha, rho, delta);
    vector[N_new] f_new = K_new' * inverse(K) * f_post;
    vector[N_new] pred = inv_logit(f_new);
    return pred;
  }
}


data {
  int<lower=1> N;
  real x[N];
  int y[N];
  real<lower=0> alpha;
  real<lower=0> rho;

  int<lower=1> N_new;
  real x_new[N_new];  
}

transformed data {
  real delta = 1e-9;
}

parameters {
  vector[N] f_tilde;
}

model {
  vector[N] f = reparam(f_tilde, x, alpha, rho, delta);

  f_tilde ~ std_normal();

  y ~ bernoulli_logit(f);
}

generated quantities {
  vector[N_new] y_new; 
  y_new = pred_binary(f_tilde, x, x_new, alpha, rho, delta);
}