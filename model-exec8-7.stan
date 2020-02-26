data {
    int N;
    int N_Pot;
    int Y[N];
    int<lower=0, upper=1> F[N];
    int<lower=1, upper=N_Pot> N2P[N];
}

parameters {
    real b[2];
    real b_I[N];
    real b_P[N_Pot];
    real<lower=0> sigma_I;
    real<lower=0> sigma_P;
}

model {
    for (n in 1:N) {
        b_I[n] ~ normal(0, sigma_I);
    }
    for (p in 1:N_Pot) {
        b_P[p] ~ normal(0, sigma_P);
    }
    for (n in 1:N) {
        Y[n] ~ poisson_log(b[1] + b[2]*F[n] + b_I[n] + b_P[N2P[n]]);
    }
}