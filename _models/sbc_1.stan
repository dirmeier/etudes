transformed data {
	real mu_sim = 0;
	real sigma_sim = 1;

	int<lower = 0> N = 10;
	vector[N] y_sim;
	for (i in 1:N)
		y_sim[i] = normal_rng(mu_sim, sigma_sim);
}

parameters {
	real mu;
	real<lower = 0> sigma;
}

model {
	mu ~ normal(0, 1);
	sigma ~ lognormal(0, 1);
  	y_sim ~ normal(mu, sigma);
}

generated quantities {
	int idsim[2] = { mu < mu_sim, sigma < sigma_sim };
}
