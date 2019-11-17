data {
    int N;
    real X[N];
    real Y[N];
}

parameters {
    real<lower=0> sigma;
    real Intercept;
    real b;
}

model {
    for (i in 1:N) {
        Y[i] ~ normal(Intercept + b * X[i], sigma);
    }
}