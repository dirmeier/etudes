data {
	int<lower=1> K;
	int<lower=1> n;
	int<lower=1> p;	
	int<lower=0,upper=1> x[n, p];
	vector<lower=0>[K] alpha;
}

parameters {    	
  ordered[p] rates[K];	
  simplex[K] pi;
}

transformed parameters {  
  vector<lower=0, upper=1>[p] prob[K];

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
    pi ~ dirichlet(alpha);
  
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
