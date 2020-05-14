data {
    int N;
    int y[N];
}

parameters {
    real b1;
    real b[N];
    real<lower=0> sigma;
}

transformed parameters {
    real q[N];
    for (n in 1:N) {
        q[n] = inv_logit(b1 + b[n]);
    }
}

model {
    for (n in 1:N) {
        b[n] ~ normal(0, sigma);
    }
    for (n in 1:N) {
        y[n] ~ binomial(8, q[n]);
    }
}

generated quantities {
    int y_pred[N];
    for (n in 1:N) {
        y_pred[n] = binomial_rng(9, q[n]);
    }
}