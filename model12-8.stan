data {
    int N;
    int I;
    real Y[N];
    int<lower=1, upper=N> To[I];
    int<lower=1, upper=N> From[I];
}

parameters {
    vector[N] r;
    real<lower=0> sigma_r;
    real<lower=0> sigma_y;
}

model {
    target += normal_lpdf(r[To] | r[From], sigma_r);
    Y ~ normal(r, sigma_y);
    sigma_y ~ normal(0, 0.1);
}