data {
    int N;
    int<lower=0, upper=1> A[N];
    real<lower=0, upper=1> Score[N];
    real<lower=0, upper=1> Weather[N];
    int<lower=0> Y[N];
}

parameters {
    real b1;
    real b2;
    real b3;
    real b4;
}

transformed parameters {
    real q[N];
    for (n in 1:N) {
        q[n] = inv_logit(b1 + b2*A[n] + b3*Score[n] + b4*Weather[n]);
    }
}

model {
    for (n in 1:N) {
        Y[n] ~ bernoulli(q[n]);
    }
}