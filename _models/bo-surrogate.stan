data {
	int<lower=1> N;	
	real x[N];
	vector[N] y;

	int<lower=1> N_star;
	real x_star[N_star];

	real<lower=0> rho;
	real<lower=0> alpha;
	real<lower=0> sigma;
}

parameters {}

model {}

generated quantities {
	vector[N_star] f_star;
	vector[N_star] f_star_cov;	

    {
      matrix[N, N] K =  cov_exp_quad(x, alpha, rho)
                         + diag_matrix(rep_vector(square(sigma), N));
      matrix[N, N] L_K = cholesky_decompose(K);

      vector[N] L_K_div_y = mdivide_left_tri_low(L_K, y);
      vector[N] K_div_y = mdivide_right_tri_low(L_K_div_y', L_K)';
      
      matrix[N, N_star] k_x_x_star = cov_exp_quad(x, x_star, alpha, rho);
      
     
      matrix[N, N_star] v_pred = mdivide_left_tri_low(L_K, k_x_x_star);
      matrix[N_star, N_star] cov_f2 = cov_exp_quad(x_star, alpha, rho) - v_pred' * v_pred
		+ diag_matrix(rep_vector(1e-10, N_star));

		f_star = (k_x_x_star' * K_div_y);
		f_star_cov = diagonal(cov_f2);
	}

}
