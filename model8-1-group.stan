data {
    int N;
    real X[N];
    real Y[N];
    int KID[N];
    int N_group;
}

parameters {
    real a[N_group];
    real b[N_group];
    real<lower=0> sigma;
}

model {
    for (n in 1:N) {
        Y[n] ~ normal(a[KID[n]] + b[KID[n]]*X[n], sigma);
    }
}

generated quantities {
    real Y_pred[N];
    for (n in 1:N) {
        Y_pred[n] = normal_rng(a[KID[n]] + b[KID[n]]*X[n], sigma);
    }
}