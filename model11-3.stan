functions {
    real ZIP_lpmf(int Y, real q, real lambda) {
        if (Y == 0) {
            return log_sum_exp(
                bernoulli_lpmf(0 | q),
                bernoulli_lpmf(1 | q) + poisson_log_lpmf(0 | lambda)
            );
        } else {
            return bernoulli_lpmf(1 | q) + poisson_log_lpmf(Y | lambda);
        }
    }
}

data {
    int N;
    int D;
    matrix[N, D] X;
    int<lower=0> Y[N];
}

parameters {
    vector[D] b[2];
}

transformed parameters {
    vector[N] q;
    vector[N] lambda;

    lambda = X * b[2];

    // ロジスティック関数によるゆるい制約
    // inv_logitはvector化に非対応
    for (n in 1:N) {
        q[n] = inv_logit((X * b[1])[n]);
    }
}

model {
    for (n in 1:N) {
        Y[n] ~ ZIP(q[n], lambda[n]);
    }
}


