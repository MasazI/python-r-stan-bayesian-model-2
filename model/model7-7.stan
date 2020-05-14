data {
    int N;
    int X[N];
    real Y[N];
}

parameters {
    real a;
    real b;
    real sigma;
    real x_true[N];
}

model {
    for (n in 1:N) {
        X[n] ~ normal(x_true[n], 2.5);
        Y[n] ~ normal(a + b*x_true[n], sigma);
    }
}