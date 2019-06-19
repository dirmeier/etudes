data {
	int<lower=1> K;
	int<lower=1> n;
	int<lower=1> p;
	int<lower=0,upper=1> x[n, p];
}

parameters {    	  	
	real<lower=0, upper=1> nu[K];	
	real<lower=0, upper=1> theta[K];
  	real<lower=0, upper=1> alpha[K];
  	real<lower=0, upper=1> beta[K];
  	real g;
}


transformed parameters {
  simplex[K] weights;  

  weights[1] = nu[1];
  for(j in 2:(K-1)) 
  {
      weights[j] = nu[j] * (1 - nu[j - 1]) * weights[j - 1] / nu[j - 1]; 
  }

  weights[K] = 1 - sum(weights[1:(K - 1)]); 
}

model {
  	real mix[K];  	
  	g ~ gamma(1, 1);
  	nu ~ beta(1, g);
  
  	// compute likelihood
  	// iterate over n rows
  	for(i in 1:n) 
  	{
  		// iterate over k cluisters
		for(k in 1:K) 
		{			
			// mixing weight from tsb
			mix[k] = log(weights[k]);
			// iterate over p columns
			for (ps in 1:p)
			{
				// likelihood if true y == 1
				real l = theta[k]       * (1 - beta[k])^x[i, ps]        * beta[k]^(1 - x[i, ps]);
				// likelihood if true y == 0
				real r = (1 - theta[k]) * (1 - alpha[k])^(1 - x[i, ps]) * alpha[k]^x[i, ps];				
				// Stan requires the log density, so we just take the log and add it to the "target" density
				mix[k] += log(l + r);
			}
		}
		target += mix;
  	}
}
