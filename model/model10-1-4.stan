data {
    int N;
    int K;
    real Age[N];
    real Sex[N];
    real Income[N];
    int<lower=1, upper=K> Y[N];
}

transformed data {
  row_vector[1] zero;
  zero = rep_row_vector(0, 1);
}

parameters {
    row_vector[K-1] b1_raw;
    row_vector[K-1] b2_raw;
    row_vector[K-1] b3_raw;
    row_vector[K-1] b4_raw;
}

transformed parameters {
    matrix[N, K] mu;
    row_vector[K] b1;
    row_vector[K] b2;
    row_vector[K] b3;
    row_vector[K] b4;
    b1 = append_col(zero, b1_raw);
    b2 = append_col(zero, b2_raw);
    b3 = append_col(zero, b3_raw);
    b4 = append_col(zero, b4_raw);
    for (n in 1:N) {
        mu[n,] = (b1 + b2 * Age[n] + b3 * Sex[n] + b4 * Income[n]);
    }
}

model {
    for (n in 1:N) {
        Y[n] ~ categorical(softmax(mu[n,]'));
    }
}