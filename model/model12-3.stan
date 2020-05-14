data {
    int N;
    vector[N] Y;
}

parameters {
    vector[N] mu;
    real<lower=0> sigma_c;
    real<lower=0> sigma_y;
}

model {
    for (n in 2:N) {
        mu[n] ~ cauchy(mu[n-1], sigma_c);
    }
    for (n in 1:N) {
        Y[n] ~ normal(mu[n], sigma_y);
    }
}