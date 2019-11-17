data {
    int N;
    real X[N];
    real Y[N];

    int N_new;
    real X_new[N_new];
}

parameters {
    real<lower=0> sigma;
    real Intercept;
    real b;
}

transformed parameters {
    real y_base[N];
    for (i in 1:N) {
        y_base[i] = Intercept + b * X[i];
    }
}

model {
    for (i in 1:N) {
        Y[i] ~ normal(y_base[i], sigma);
    }
}

generated quantities {
    real y_base_new[N_new];
    real y_new[N_new];
    for (i in 1:N_new) {
        y_base_new[i] = Intercept + b * X_new[i];
        y_new[i] = normal_rng(y_base_new[i], sigma);
    }
}