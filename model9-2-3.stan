data {
    int N;
    int T;
    real Time[T];
    matrix[N, T] Y;

    int T_new;
    real Time_new[T_new];
}

parameters {
    vector[N] log_a;
    vector[N] log_b;

    real a0;
    real b0;
    real<lower=0> sigma_a;
    real<lower=0> sigma_b;
    real<lower=0> sigma;
}

transformed parameters {
    vector[N] a;
    vector[N] b;
    matrix[N, T] mu;

    a = exp(log_a);
    b = exp(log_b);

    for (t in 1:T) {
        for (n in 1:N) {
            mu[n, t] = a[n]*(1 - exp(-b[n].*Time[t]));
        }
    }
}

model {
    log_a ~ normal(a0, sigma_a);
    log_b ~ normal(b0, sigma_b);

    to_vector(Y) ~ normal(to_vector(mu), sigma);
}

generated quantities {
    real y_new[N, T_new];
    for (n in 1:N) {
        for (t in 1:T_new) {
            y_new[n, t] = normal_rng(a[n]*(1 - exp(-b[n]*Time_new[t])), sigma);
        }
    }
}