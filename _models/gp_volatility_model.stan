functions {
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
}


transformed data {
    matrix[N, Q] Pmat;

    for (j in 1:Q) {
        Pmat[, j] = phi(times, L, j);
    }
}

parameters {
    vector[Q] beta_spd;
    real<lower=0> alpha;
    real<lower=0> rho;
    real gamma;
    real<lower=0> sigma_sigma;
    vector<lower=0>[N] sigmas;
}


transformed parameters {
    vector[N] f;
    {
        vector[Q] spd_diag;
        for(j in 1:Q) {
            spd_diag[j] = sqrt(S(sqrt(lambda(L, j)), alpha, rho));
        }
        f = Pmat * (spd_diag .* beta_spd);
    }
}


model {
    beta_spd ~ std_normal();
    alpha ~ std_normal();
    rho ~ inv_gamma(5, 5);
    gamma ~ normal(0, 5);

    sigma_sigma  ~ std_normal();
    sigmas ~ normal(exp(f + gamma), sigma_sigma);

    y ~ normal(0, sigmas);
}
