data {
	int<lower=1> K;
	int<lower=1> N;	
	int x[N];	
}

parameters {    	
  ordered[K] mu;
  	
	real<lower=0> shape;
	real<lower=0, upper=1> stick[K];
}

transformed parameters {  
  	simplex[K] weights;

  	weights[1] = stick[1];
  	for(j in 2:(K-1)) 
  	{
      weights[j] = stick[j] * (1 - stick[j - 1]) * weights[j - 1] / stick[j - 1]; 
  	}
	  weights[K] = 1 - sum(weights[1:(K - 1)]);
}

model {
  	real mix[K];
    mu ~ uniform(1, 50);

  	shape ~ gamma(1, 1);
  	stick ~ beta(1, shape);	
	  
  	for(i in 1:N) 
  	{
  		for(k in 1:K) 
  		{
  			mix[k] = log(weights[k]) + poisson_lpmf(x[i] | mu[k]);
  		}
		  target += log_sum_exp(mix);
  	}
}
