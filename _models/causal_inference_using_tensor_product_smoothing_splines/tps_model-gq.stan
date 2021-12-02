#include causal_inference_using_gps_tensor_splines.stan

generated quantities {
    vector[N_I] ite;
    {
        real      tr1[N_I]   = to_array_1d(tr + 1);
        real      ur[N_I]    = to_array_1d(U[i_to_o]);
        vector[2] uxr[N_I]   = concat_rr(N_I, ur, xr);
        vector[3] uxtr1[N_I] = concat_vr(N_I, uxr, tr1);

        matrix[N_I, n_coefs * n_coefs * n_coefs] y_star_design = tensor_spline(uxtr1, n_coefs, order, knots);

        vector[N_I] y_star_mu  = y_star_design * y_beta;

        for (i in 1:N_I)
            ite[i] = normal_rng(y_star_mu[i] - y[i], y_scale);
    }
}
