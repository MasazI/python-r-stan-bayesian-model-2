data {
    int N;
    real<lower=0> Y[N];
    real<lower=0> Area[N];
    int N_new;
    real<lower=0> Area_new[N_new];
}

parameters {
    real b1;
    real b2;
    real<lower=0> sigma;
}

transformed parameters {
    real mu[N];
    for (n in 1:N) {
        mu[n] = b1 + b2 * Area[n];
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
        Y_new[n] = normal_rng(b1 + b2*Area_new[n], sigma);
    }
}