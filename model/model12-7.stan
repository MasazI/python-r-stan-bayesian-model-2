data {
    int I;
    int J;
    real Y[I, J];
    int T;
    int<lower=0, upper=T> TID[I, J];
}

parameters {
    real r[I, J];
    vector[T] beta;
    real<lower=0> sigma_r;
    real<lower=0> sigma_b;
    real<lower=0> sigma_y;
}

model {
    for (i in 1:I) {
        for (j in 3:J) {
            target += normal_lpdf(r[i,j] | 2*r[i,j-1] - r[i,j-2], sigma_r);
        }
    }
    for (i in 3:I) {
        for (j in 1:J) {
            target += normal_lpdf(r[i,j] | 2*r[i-1,j] - r[i-2,j], sigma_r);
        }
    }

    beta ~ student_t(6, 0, sigma_b);

    for (i in 1:I) {
        for (j in 1:J) {
            Y[i, j] ~ normal(beta[TID[i, j]] + r[i, j], sigma_y);
        }
    }
}