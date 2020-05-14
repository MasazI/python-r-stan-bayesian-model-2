data {
    int I;
    int Y[I];
}

parameters {
    vector[I] r;
    real<lower=0> sigma_r;
}

model {
    target += normal_lpdf(r[2:I] | r[1:(I-1)], sigma_r);
    Y ~ poisson_log(r);
}

generated quantities {
    vector[I] Y_mean;
    Y_mean = exp(r);
}
