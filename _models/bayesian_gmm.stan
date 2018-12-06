data {
	int<lower=0> n;
	int<lower=1> p;
	row_vector[p] x[n];
	vector<lower=0>[3] alpha;
}

parameters {
	ordered[p] mu[3];
	simplex[3] theta;
	cholesky_factor_corr[p] L;
}

model {	
	real mix[3];
	L ~ lkj_corr_cholesky(5);
	

	theta ~ dirichlet(alpha);
	for (i in 1:3) 
	{
		mu[i] ~ normal(0, 5);
	}

 	for (i in 1:n) 
 	{
 		for (k in 1:3) 
 		{
 			mix[k] = log(theta[k]) + multi_normal_cholesky_lpdf(x[i] | mu[k], L);
 		}
 		target += log_sum_exp(mix);
 	}
}
