functions {
    vector[] concat_rr(int N, real[] arr1, real[] arr2) {
        vector[2] v[N];
        v[:, 1] = arr1;
        v[:, 2] = arr2;
        return v;
    }

    vector[] concat_vr(int N, vector[] arr1, real[] arr2) {
        int P1 = size(arr1[1]);        
        vector[P1 + 1] v[N];
        for (i in 1:P1) v[:, i] = arr1[:, i];        
        v[:, P1 + 1] = arr2;
        return v;
    }


    row_vector kronecker(row_vector a, row_vector b) {
        int n = size(a);
        int m = size(b);
        int idx;

        row_vector[n * m] r;
        for (i in 1:m) {
            idx = (i - 1) * n + 1;
            r[idx:(idx + n - 1)] = a * b[i];  
        }

        return r;
    }

    row_vector kronecker_recursive(matrix[] B, int P, int idx) {
        
        if (P == 2) 
            return kronecker(B[1, idx], B[2, idx]);
        else 
            return kronecker(kronecker(B[1, idx], B[2, idx]), B[3, idx]);
    }


    real bspline_basis(real x, vector knots, int i, int d)
    {
      int n_knots = size(knots);
      real b1 = 0;
      real b2 = 0;

      if (d == 0)
        return knots[i + 1] <= x && x < knots[i + 2] ? 1.0 : 0.0;

      if (knots[i + d + 1] != knots[i + 1])
        b1 = (x - knots[i + 1]) / (knots[i + d + 1] - knots[i + 1])
          * bspline_basis(x, knots, i, d - 1);

      if (knots[i + d + 2] == knots[i + 2])
        b2 = (knots[i + d + 2] - x) / (knots[i + d + 2] - knots[i + 2])
          * bspline_basis(x, knots, i + 1, d - 1);

      return b1 + b2;
    }

    matrix bspline(vector u, int n_coef, int d, vector knots)
    {
        int n = size(u);
        matrix[n, n_coef] mu; 
              
        for (i in 1:n) {            
            for (j in 1:n_coef) {
                mu[i, j] = bspline_basis(u[i], knots, j - 1, d);                  
            }
        }  

        return mu;
    }


    matrix tensor_spline(vector[] u, int n_coefs, int d, vector[] knots)
    {        
        int NP[2] = dims(u);
        int N = NP[1];
        int P = NP[2];
        int new_dim = 1;
        for (i in 1:P) new_dim *= n_coefs;
        
        matrix[N, new_dim] mu;
        matrix[N, n_coefs] B[P];
        for (i in 1:P) B[i] = bspline(to_vector(u[:, i]), n_coefs, d, knots[i]);
    
        for (i in 1:N) {            
            mu[i, :] = kronecker_recursive(B, P, i);
        }  

        return mu;
    }
}

data {
    int<lower=0> N_I;
    int<lower=0> N_O;    
    int<lower=0, upper=N_O> i_to_o[N_I];
    vector[N_I] X;
    vector[N_I] tr;
    vector[N_I] y;  
    
    real<lower=0> alpha;
    real<lower=0> beta;
    int<lower=1> order;
    int<lower=1> n_coefs;    

    vector[n_coefs + order + 1] u_knots;
    vector[n_coefs + order + 1] x_knots;
    vector[n_coefs + order + 1] tr_knots;
}

transformed data {    
    real xr[N_I]  = to_array_1d(X);
    real trr[N_I] = to_array_1d(tr);
    vector[n_coefs + order + 1] knots[3];    
    knots[1] = u_knots;
    knots[2] = x_knots;
    knots[3] = tr_knots;
}


parameters {
    real<lower=0> u_scale;
    vector[N_O]   U_tilde;

    real          x_beta;
    real<lower=0> x_scale;
    
    vector[n_coefs * n_coefs] t_beta;    
    real<lower=0>             t_scale;
        
    vector[n_coefs * n_coefs * n_coefs] y_beta;    
    real<lower=0>                       y_scale;    
}


transformed parameters {
    vector[N_O] U = U_tilde * u_scale;
}

model {
    real      ur[N_I]   = to_array_1d(U[i_to_o]);
    vector[2] uxr[N_I]  = concat_rr(N_I, ur, xr);
    vector[3] uxtr[N_I] = concat_vr(N_I, uxr, trr);

    matrix[N_I, n_coefs * n_coefs] t_design  = tensor_spline(uxr,  n_coefs, order, knots);
    matrix[N_I, n_coefs * n_coefs * n_coefs] y_design  = tensor_spline(uxtr, n_coefs, order, knots);    
    

    u_scale         ~ inv_gamma(alpha, beta);
    U_tilde         ~ std_normal();

    x_beta          ~ std_normal();
    x_scale         ~ std_normal();

    t_beta          ~ std_normal();
    t_scale         ~ std_normal();

    y_beta          ~ normal(0, 1);    
    y_scale         ~ std_normal();
        
    xr  ~ normal(U[i_to_o], x_scale);
    tr  ~ normal(t_design * t_beta, t_scale);
    y   ~ normal(y_design * y_beta, y_scale);
}

