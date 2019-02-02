data {
	int<lower=0> K;
	int<lower=0> n;
	int<lower=1> p;
	real<lower=0> a;
	real<lower=0> b;
	int<lower=0,upper=1> x[n, p];
	real alpha;
}

parameters {    	
  	ordered[K] rates;
	real <lower=0, upper=1> nu[K];	
}

transformed parameters {
  simplex[K] pi;
  vector<lower=0, upper=1>[K] prob;

  pi[1] = nu[1];
  for(j in 2:(K-1)) 
  {
      pi[j] = nu[j] * (1 - nu[j - 1]) * pi[j - 1] / nu[j - 1]; 
  }

  pi[K] = 1 - sum(pi[1:(K - 1)]);
  prob = inv_logit(rates);  
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
				mix[k] += bernoulli_lpmf(x[i, ps] | prob[k]);
			}
		}
		target += log_sum_exp(mix);
  	}
}
