data {
    int N;
    vector[N] X;
    vector[N] Y;
    int N_group;
    int<lower=1, upper=N_group> KID[N];
}

parameters {
    real a0;
    real b0;

    vector[N_group] a;
    vector[N_group] b;

    real<lower=0> sigma_a;
    real<lower=0> sigma_b;
    real<lower=0> sigma;
}

model {
    a ~ normal(a0, sigma_a);
    b ~ normal(b0, sigma_b);
    Y ~ normal(a[KID] + b[KID] .* X, sigma);
}