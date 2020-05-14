data {
    int N;
    int D;
    vector[D] Y[N];
}

parameters {
    vector[D] mu;
    cov_matrix[D] cov;
}

model {
    for (n in 1:N) {
        Y[n] ~ multi_normal(mu, cov);
    }
}