data {
    int y1_N;
    int y2_N;
    real y1[y1_N];
    real y2[y2_N];
}

parameters {
    real<lower=0> sigma;
    real mu1;
    real mu2;
}

model {
    for (i in 1:y1_N) {
        y1[i] ~ normal(mu1, sigma);
    }
    for (i in 1:y2_N) {
        y2[i] ~ normal(mu2, sigma);
    }
}