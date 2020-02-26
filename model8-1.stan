data {
    int N;
    real X[N];
    real Y[N];

    int N_new;
    real X_new[N_new];
}

parameters {
    real a;
    real b;
    real<lower=0> sigma;
}

model {
    for (n in 1:N) {
        Y[n] ~ normal(a + b*X[n], sigma);
    }
}

generated quantities {
    real Y_pred[N];
    real Y_new[N_new];
    for (n in 1:N) {
        Y_pred[n] = normal_rng(a + b*X[n], sigma);
    }
    for (n in 1:N_new) {
        Y_new[n] = normal_rng(a + b*X_new[n], sigma);
    }
}