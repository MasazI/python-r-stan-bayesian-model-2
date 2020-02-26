data {
    int N;
    real Y[N];
    real X[N];
    int N_group;
    int N_industry;
    int<lower=1,upper=N_group> KID[N];
    int<lower=1,upper=N_industry> K2G[N_group];
}

parameters {
    real a0;
    real a1[N_industry];
    real a[N_group];

    real b0;
    real b1[N_industry];
    real b[N_group];

    real<lower=0> sigma_a1;
    real<lower=0> sigma_b1;
    real<lower=0> sigma_a;
    real<lower=0> sigma_b;
    real<lower=0> sigma;
}

model {
    for (n in 1:N_industry) {
        a1[n] ~ normal(a0, sigma_a1);
        b1[n] ~ normal(b0, sigma_b1);
    }
    for (n in 1:N_group) {
        a[n] ~ normal(a1[K2G[n]], sigma_a);
        b[n] ~ normal(b1[K2G[n]], sigma_b);
    }
    for (n in 1:N) {
        Y[n] ~ normal(a[KID[n]] + b[KID[n]]*X[n], sigma);
    }
}