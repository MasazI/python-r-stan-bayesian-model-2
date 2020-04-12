data {
    int N;
    vector[N] X;
    vector[N] Y;

    int N_new;
    vector[N_new] X_new;
}

parameters {
    real<lower=0> sigma;
    real Intercept;
    real b;
}

transformed parameters {
    vector[N] y_base;
    y_base = Intercept + b * X;
}

model {
    Y ~ normal(y_base, sigma);
}

generated quantities {
    vector[N_new] y_base_new;
    vector[N_new] y_new;
    for (n in 1:N_new) {
        y_base_new[n] = Intercept + b * X_new[n];
        y_new[n] = normal_rng(y_base_new[n], sigma);
    }
}