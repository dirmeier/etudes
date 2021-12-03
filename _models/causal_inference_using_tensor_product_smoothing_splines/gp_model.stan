#include functions.stan

data {
    int<lower=0> N_I;
    int<lower=0> N_O;
    int<lower=0, upper=N_O> i_to_o[N_I];
    vector[N_I] X;
    vector[N_I] tr;
    vector[N_I] y;
    real alpha;
    real beta;
}

transformed data {
    real xr[N_I] = to_array_1d(X);
    real trr[N_I] = to_array_1d(tr);

}

parameters {
    real<lower=0>        u_scale;
    vector[N_O]          U_tilde;

    real<lower=0>        cov_x_rho;
    real<lower=0>        cov_x_scale;
    real<lower=0>        x_scale;

    real<lower=0>        cov_t_rhos[2];
    real<lower=0>        cov_t_scale;
    real<lower=0>        t_scale;

    real<lower=0>        cov_y_rhos[3];
    real<lower=0>        cov_y_scale;
    real<lower=0>        y_scale;
}


transformed parameters {
    vector[N_O] U = U_tilde * u_scale;
}

model {
    real      ur[N_I]   = to_array_1d(U[i_to_o]);
    vector[2] uxr[N_I]  = concat_rr(N_I, ur, xr);
    vector[3] uxtr[N_I] = concat_vr(N_I, uxr, trr);

    matrix[N_I, N_I] KX = gp_exp_quad_cov(ur, ur, cov_x_scale, cov_x_rho)
        + diag_matrix(rep_vector(1e-10, N_I))
        + diag_matrix(rep_vector(square(x_scale), N_I));
    matrix[N_I, N_I] LX = cholesky_decompose(KX);

    matrix[N_I, N_I] KT = gp_exp_quad_cov(uxr, uxr, cov_t_scale, cov_t_rhos)
        + diag_matrix(rep_vector(1e-10, N_I))
        + diag_matrix(rep_vector(square(t_scale), N_I));
    matrix[N_I, N_I] LT = cholesky_decompose(KT);

    matrix[N_I, N_I] KY = gp_exp_quad_cov(uxtr, uxtr, cov_y_scale, cov_y_rhos)
        + diag_matrix(rep_vector(1e-10, N_I))
        + diag_matrix(rep_vector(square(y_scale), N_I));
    matrix[N_I, N_I] LY = cholesky_decompose(KY);


    u_scale                        ~ inv_gamma(alpha, beta);
    U_tilde                        ~ std_normal();

    cov_x_rho                      ~ inv_gamma(alpha, beta);
    cov_x_scale                    ~ std_normal();
    x_scale                        ~ std_normal();

    cov_t_rhos                     ~ inv_gamma(alpha, beta);
    cov_t_scale                    ~ std_normal();
    t_scale                        ~ std_normal();

    cov_y_rhos                     ~ inv_gamma(alpha, beta);
    cov_y_scale                    ~ std_normal();
    y_scale                        ~ std_normal();

    X   ~ multi_normal_cholesky(rep_vector(0, N_I), LX);
    tr  ~ multi_normal_cholesky(rep_vector(0, N_I), LT);
    y   ~ multi_normal_cholesky(rep_vector(0, N_I), LY);
}
