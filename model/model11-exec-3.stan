data {
    int N;
    int<lower=0, upper=1> Y[N];
}

parameters {
    real<lower=0, upper=1> q;
}

model {
    for (n in 1:N) {
        vector[4] lp;
        lp[1] = log(0.5) + log(0.4) + bernoulli_lpmf(Y[n] | q);
        lp[2] = log(0.5) + log(0.6) + bernoulli_lpmf(Y[n] | 1);
        lp[3] = log(0.5) + log(0.4) + bernoulli_lpmf(Y[n] | 1);
        lp[4] = log(0.5) + log(0.6) + bernoulli_lpmf(Y[n] | 1);
        target += log_sum_exp(lp);
    }
}