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

    for (i in 1:P1) {
      v[:, i] = arr1[:, i];
    }
    v[:, P1 + 1] = arr2;

    return v;
  }

  real bspline_basis(real x, vector knots, int i, int d)
  {
    int n_knots = size(knots);
    real b1 = 0;
    real b2 = 0;

    if (d == 0) {
      return knots[i] <= x && x < knots[i + 1] ? 1.0 : 0.0;
    }

    if (knots[i + d] != knots[i]) {
      b1 = (x - knots[i]) / (knots[i + d] - knots[i]);
      b1 = b1 * bspline_basis(x, knots, i, d - 1);
    }

    if (knots[i + d + 1] != knots[i + 1]) {
      b2 = (knots[i + d + 1] - x) / (knots[i + d + 1] - knots[i + 1]);
      b2 = b2 * bspline_basis(x, knots, i + 1, d - 1);
    }

    return b1 + b2;
  }

  matrix bspline(vector u, int n_coef, int d, vector knots)
  {
    int n = size(u);
    matrix[n, n_coef] mu;

    if (n_coef + d + 1 != size(knots)) {
      reject("n_coef + d != size(knots)");
    }

    for (i in 1:n) {
      for (j in 1:n_coef) {
        mu[i, j] = bspline_basis(u[i], knots, j, d);
      }
    }

    return mu;
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
    if (P == 2) {
      return kronecker(B[1, idx], B[2, idx]);
    }
    else {
      return kronecker(kronecker(B[1, idx], B[2, idx]), B[3, idx]);
    }
  }

  matrix tensor_spline(vector[] u, int n_coefs, int d, vector[] knots)
  {
    int NP[2] = dims(u);
    int N = NP[1];
    int P = NP[2];
    int new_dim = 1;

    for (i in 1:P) {
      new_dim *= n_coefs;
    }

    matrix[N, new_dim] mu;
    matrix[N, n_coefs] B[P];

    for (i in 1:P) {
      B[i] = bspline(to_vector(u[:, i]), n_coefs, d, knots[i]);
    }

    for (i in 1:N) {
      mu[i, :] = kronecker_recursive(B, P, i);
    }

    return mu;
  }
}
