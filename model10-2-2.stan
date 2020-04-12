data {
    int N;
    real X[N];
    real Y[N];
    int N_group;
    int<lower=1, upper=N_group> KID[N];
}

parameters {
    real a0;
    real ak[N_group];
    real<lower=0> sigma_a;
    real b0;
    real bk[N_group];
    real<lower=0> sigma_b;
    real<lower=0> sigma;
}

transformed parameters {
    real a[N_group];
    real b[N_group];
    for (n in 1:N_group) {
        a[n] = a0 + ak[n];
        b[n] = b0 + bk[n];
    }
}

model {
    sigma_a ~ student_t(4, 0, 100);

    for (n in 1:N_group) {
        ak[n] ~ normal(0, sigma_a);
        bk[n] ~ normal(0, sigma_b);
    }
    for (n in 1:N) {
        Y[n] ~ normal(a[KID[n]] + b[KID[n]]*X[n], sigma);
    }
}

generated quantities {
    real Y_pred[N];
    real ak_pred[N_group];
    real bk_pred[N_group];
    for (n in 1:N_group) {
        ak_pred[n] = normal_rng(0, sigma_a);
        bk_pred[n] = normal_rng(0, sigma_b);
    }
    for (n in 1:N) {
        Y_pred[n] = normal_rng(a[KID[n]] + b[KID[n]]*X[n], sigma);
    }
}
