functions {
    vector warp(vector f, real[] beta, real[] gamma, int K) {
        vector[size(f)] y = rep_vector(0, size(f));
        for (i in 1:K) {
            y += (beta[i] ^ 2) * log(exp(f + gamma[i]) + 1.0);
        }
        return y;
    }

    real lambda(real L, int j) {
        return ((j * pi()) / (2 * L)) ^ 2;
    }

    real S(real w, real alpha, real rho) {
        return alpha^2 * sqrt(2 * pi()) * rho * exp(-0.5 * (rho * w)^2);
    }

    vector phi(vector x, real L, int j) {
        return 1 / sqrt(L) * sin(j  * pi()/(2 * L) * (x + L));
    }
}

data {
    int<lower=1> N;
    vector[N] times;
    vector[N] y;
    int<lower=1> Q;
    real<lower=0> L;
    int<lower=1> K;
}


transformed data {
    matrix[N, Q] Pmat;

    for (j in 1:Q) {
        Pmat[, j] = phi(times, L, j);
    }
}

parameters {
    vector[Q] beta_spd;

    real<lower=0> beta[K];
    real<lower=0> gamma[K];
    real<lower=0> rho;
}


transformed parameters {
    vector[N] f;
    vector<lower=0>[N] sigmas;
    {
        vector[Q] spd_diag;
        for(j in 1:Q) {
            spd_diag[j] = sqrt(S(sqrt(lambda(L, j)), 1.0, rho));
        }
        f = Pmat * (spd_diag .* beta_spd);
    }

   sigmas = warp(f, beta, gamma, K);
}

model {
    beta_spd ~ std_normal();
    beta ~ std_normal();
    gamma ~ inv_gamma(5, 5);
    rho ~ inv_gamma(5, 5);

    y ~ normal(0, sigmas);
}

generated quantities {
    vector[N] y_hat;
    for (i in 1:N)
        y_hat[i] = normal_rng(0, sigmas[i]);
}
