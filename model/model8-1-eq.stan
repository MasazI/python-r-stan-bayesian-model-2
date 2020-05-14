data {
    int N;
    real X[N];
    real Y[N];
    int N_group;
    int<lower=1, upper=N_group> KID[N];
}

parameters {
    real a0;
    real b0;

    real a[N_group];
    real b[N_group];

    real<lower=0> sigma_a;
    real<lower=0> sigma_b;
    real<lower=0> sigma;
}

model {
    for (n in 1:N_group){
        a[n] ~ normal(a0, sigma_a);
        b[n] ~ normal(b0, sigma_b);
    }
    for (n in 1:N) {
        Y[n] ~ normal(a[KID[n]] + b[KID[n]]*X[n], sigma);
    }
}