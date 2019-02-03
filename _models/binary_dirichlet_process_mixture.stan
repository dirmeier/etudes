data {
	int<lower=1> K;
	int<lower=1> n;
	int<lower=1> p;
	real<lower=0> a;
	real<lower=0> b;
	int<lower=0,upper=1> x[n, p];
	real<lower=1> alpha;
}

parameters {    	
  	ordered[p] rates[K];
	real<lower=0, upper=1> nu[K];	
}

transformed parameters {
  simplex[K] pi;
  vector<lower=0, upper=1>[p] prob[K];

  pi[1] = nu[1];
  for(j in 2:(K-1)) 
  {
      pi[j] = nu[j] * (1 - nu[j - 1]) * pi[j - 1] / nu[j - 1]; 
  }

  pi[K] = 1 - sum(pi[1:(K - 1)]);
  for (k in 1:K) 
  {
  	for (ps in 1:p) 
  	{
  		prob[k, ps] = inv_logit(rates[k, ps]);  
  	}
  }   
}

model {
  	real mix[K];
  	nu ~ beta(1, alpha);	
  
  	for(i in 1:n) 
  	{
		for(k in 1:K) 
		{
			mix[k] = log(pi[k]);
			for (ps in 1:p)
			{
				mix[k] += bernoulli_lpmf(x[i, ps] | prob[k, ps]);
			}
		}
		target += log_sum_exp(mix);
  	}
}
