data {
    int N;
    real X[N];
    real<lower=0> Y[N];
    int N_new;
    real X_new[N_new];
}

parameters {
    real b1;
    real b2;
    real<lower=0, upper=30> x0;
    real<lower=0> sigma;
}

transformed parameters {
    real mu[N];
    for (n in 1:N){
        mu[n] = b1 + b2 * (X[n] - x0)^2;
    }
}

model {
    for (n in 1:N) {
        Y[n] ~ normal(mu[n], sigma);
    }
}

generated quantities {
    real Y_pred[N];
    real Y_new[N_new];

    for (n in 1:N) {
        Y_pred[n] = normal_rng(mu[n], sigma);
    }
    for (n in 1:N_new) {
        Y_new[n] = normal_rng(b1 + b2 * (X_new[n] - x0)^2, sigma);
    }
}