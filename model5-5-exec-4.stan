data {
    int N;
    int<lower=0, upper=1> A[N];
    real<lower=0, upper=1> Score[N];
    int<lower=1, upper=3> Weather[N];
    int<lower=0> Y[N];
}

parameters {
    real b[3];
    real bw1;
    real bw2;
    real bw3;
}

transformed parameters {
    real bw[3];
    real q[N];

    bw[1] = bw1;
    bw[2] = bw2;
    bw[3] = bw3;
    for (n in 1:N){
        q[n] = b[1] + b[2]*A[n] + b[3]*Score[n] + bw[Weather[n]];
    }

}

model {
    for (n in 1:N) {
        Y[n] ~ bernoulli_logit(q[n]);
    }
}