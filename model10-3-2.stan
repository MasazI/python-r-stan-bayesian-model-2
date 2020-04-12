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

    vector[N_group] a_raw;
    vector[N_group] b_raw;

    real<lower=0> sigma_a;
    real<lower=0> sigma_b;
    real<lower=0> sigma;
}

transformed parameters {
    vector[N_group] a;
    vector[N_group] b;
    a = a0 + sigma_a*a_raw;
    b = b0 + sigma_b*b_raw;
}

model {
    a_raw ~ normal(0, 1);
    b_raw ~ normal(0, 1);
    Y ~ normal(a[KID] + b[KID] .* X, sigma);
}