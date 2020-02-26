data {
    int N_obs;
    int N_cens;
    real Y_obs[N_obs];
    real L;
}

parameters {
    real mu;
    real<lower=0> sigma;
}

model {
    for (n in 1:N_obs) {
        Y_obs[n] ~ normal(mu, sigma);
    }
    target += N_cens * normal_lcdf(L | mu, sigma);
}