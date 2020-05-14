data {
    int N;
    int T;
    real Time[T];
    real Y[N, T];

    int T_new;
    real Time_new[T_new];
}

parameters {
    real log_a[N];
    real log_b[N];

    real a0;
    real b0;
    real<lower=0> sigma_a;
    real<lower=0> sigma_b;
    real<lower=0> sigma;
}

transformed parameters {
    real a[N];
    real b[N];
    for (n in 1:N) {
        a[n] = exp(log_a[n]);
        b[n] = exp(log_b[n]);
    }
}

model {
    for (n in 1:N) {
        log_a[n] ~ normal(a0, sigma_a);
        log_b[n] ~ normal(b0, sigma_b);
    }

    for (n in 1:N) {
        for (t in 1:T) {
            Y[n, t] ~ normal(a[n]*(1 - exp(-b[n]*Time[t])), sigma);
        }
    }
}

generated quantities {
    real y_new[N, T_new];
    for (n in 1:N) {
        for (t in 1:T_new) {
            y_new[n, t] = normal_rng(a[n]*(1 - exp(-b[n]*Time_new[t])), sigma);
        }
    }
}