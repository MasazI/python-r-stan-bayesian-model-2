data {
    int N;
    real Y[N];
    real Weight[N];
    int Age[N];

    int N_new;
    real Age_new[N_new];
}

parameters {
    real c1;
    real c2;
    real<lower=0> sigma_c;
    real b1;
    real b2;
    real b3;
    real<lower=0> sigma_b;
}

model {
    for (n in 1:N) {
        Weight[n] ~ normal(c1 + c2*Age[n], sigma_c);
        Y[n] ~ normal(b1 + b2*Age[n] + b3*Weight[n], sigma_b);
    }
}

generated quantities {
    real Weight_new[N_new];
    real Y_new[N_new];
    for (n in 1:N_new) {
        Weight_new[n] = normal_rng(c1 + c2*Age_new[n], sigma_c);
        Y_new[n] = normal_rng(b1 + b2*Age_new[n] + b3*Weight_new[n], sigma_b);
    }
}