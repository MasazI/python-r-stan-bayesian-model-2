data {
    int N;
    int<lower=0> Y[N];
}

parameters {
    real<lower=0> lambda;
}

model {
    for (n in 1:N) {
        vector[40-Y[n]+ 1] k;
        for (i in Y[n]:40) {
            k[i-Y[n]+1] = poisson_lpmf(i | lambda) + binomial_lpmf(Y[n] | i, 0.5);
        }
        target += log_sum_exp(k);
    }
}