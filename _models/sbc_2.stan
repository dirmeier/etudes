data {
	int<lower=1> n;
	vector[n] x;
}

transformed data {
	real beta_sim = normal_rng(0, 1);
	real<lower=0> sigma_sim = lognormal_rng(0, 1);
	
	vector[n] y_sim;
	for (i in 1:n)
		y_sim[i] = normal_rng(x[i] * beta_sim, sigma_sim);
}

parameters {
	real beta;
	real<lower = 0> sigma;
}

model {
	beta ~ normal(0, 10);
	sigma ~ lognormal(0, 5);
  	y_sim ~ student_t(10, x * beta, sigma);
}

generated quantities {
	int idsim[2] = { beta < beta_sim, sigma < sigma_sim };
}
