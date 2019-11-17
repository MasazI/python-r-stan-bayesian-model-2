data {
    int y1_N;
    int y2_N;
    real y1[y1_N];
    real y2[y2_N];
}

parameters {
    real<lower=0> sigma1;
    real<lower=0> sigma2;
    real mu1;
    real mu2;
}

model {
    for (i in 1:y1_N) {
        y1[i] ~ normal(mu1, sigma1);
    }
    for (i in 1:y2_N) {
        y2[i] ~ normal(mu2, sigma2);
    }
}