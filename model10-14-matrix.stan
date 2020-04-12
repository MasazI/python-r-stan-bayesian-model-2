data {
    int N;
    int D;
    int K;
    matrix[N, D] X;
    int<lower=1, upper=K> Y[N];
}

transformed data {
  vector[D] zero;
  zero = rep_vector(0, D);
}

parameters {
    matrix[D, K-1] b_raw;
}

transformed parameters {
    matrix[N, K] mu;
    matrix[D, K] b;
    b = append_col(zero, b_raw);
    mu = X * b;
}

model {
    for (n in 1:N) {
        Y[n] ~ categorical(softmax(mu[n,]'));
    }
}