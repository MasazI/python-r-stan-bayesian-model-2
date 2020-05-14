data {
    int N;
    int<lower=0> y[N];
    real<lower=0> x[N];
    int<lower=0, upper=1> f[N];
}

parameters {
    real b[3];
}

transformed parameters {
    real lambda[N];
    for (n in 1:N) {
        lambda[n] = b[1] + b[2]*x[n] + b[3]*f[n];
    }
}

model {
    for (n in 1:N) {
        y[n] ~ poisson_log(lambda[n]);
    }
}

generated quantities {
    int y_pred[N];
    for (n in 1:N) {
        y_pred[n] = poisson_log_rng(lambda[n]);
    }
}