data {
	int<lower=0> K;
	int<lower=0> n;
	int<lower=1> p;
	row_vector[p] x[n];
	real alpha;
}

parameters {    	
  	ordered[p] mu[K];
	cholesky_factor_corr[p] L;
	real <lower=0, upper=1> nu[K];
}

transformed parameters {
  simplex[K] pi;
  pi[1] = nu[1];
  for(j in 2:(K-1)) 
  {
      pi[j] = nu[j] * (1 - nu[j - 1]) * pi[j - 1] / nu[j - 1]; 
  }

  pi[K] = 1 - sum(pi[1:(K - 1)]);
}

model {
  	real mix[K];

  	L ~ lkj_corr_cholesky(5);
  	nu ~ beta(1, alpha);	
	for (i in 1:K) 
	{
		mu[i] ~ normal(0, 5);
	}

  
  	for(i in 1:n) 
  	{
		for(k in 1:K) 
		{
			mix[k] = log(pi[k]) + multi_normal_cholesky_lpdf(x[i] | mu[k], L);
		}
		target += log_sum_exp(mix);
  	}
}
