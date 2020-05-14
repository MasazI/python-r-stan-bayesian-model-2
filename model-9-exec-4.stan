data {
    int N;
    vector[N] Y;
    vector[N] X;
    int N_group;
    int N_industry;
    int<lower=1,upper=N_group> KID[N];
    int<lower=1,upper=N_industry> K2G[N_group];
}

parameters {
    real a0;
    vector[N_industry] a1;
    vector[N_group] a;

    real b0;
    vector[N_industry] b1;
    vector[N_group] b;

    real<lower=0> sigma_a1;
    real<lower=0> sigma_b1;
    real<lower=0> sigma_a;
    real<lower=0> sigma_b;
    real<lower=0> sigma;
}

model {
    a1 ~ normal(a0, sigma_a1);
    b1 ~ normal(b0, sigma_b1);
    a ~ normal(a1[K2G], sigma_a);
    b ~ normal(b1[K2G], sigma_b);
    Y ~ normal(a[KID] + b[KID] .* X, sigma);
}