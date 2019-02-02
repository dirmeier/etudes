data {
	int<lower=0> K;
	int<lower=0> n;
	int<lower=1> p;
	real a;
	real b;
	int x[n, p];
	real alpha;
}

parameters {    	
  	ordered[p] prob[K];
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
  	nu ~ beta(1, alpha);	
	for (i in 1:K) 
	{
		prob[i] ~ beta(a, b);
	}

  
  	for(i in 1:n) 
  	{
		for(k in 1:K) 
		{
			mix[k] = log(pi[k]);
			for (ps in 1:p)
			{
				mix[k] += bernoulli_lpmf(x[i,p] | prob[k, ps]);
			}
		}
		target += log_sum_exp(mix);
  	}
}
