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
    Y ~ multi_normal(mu, cov);
}