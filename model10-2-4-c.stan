data {
    int N;
    vector[N] X;
    vector[N] Y;
    int N_group;
    int<lower=1, upper=N_group> KID[N];
    real Nu;
}

parameters {
    vector[2] ab0;
    vector[2] ab[N_group];
    real<lower=0> sigma;

    cholesky_factor_corr[2] corr_chol;
    vector<lower=0>[2] sigma_vec;
}

transformed parameters {
    vector[N_group] a;
    vector[N_group] b;

    cholesky_factor_cov[2] cov_chol;

    for (n in 1:N_group) {
        a[n] = ab[n, 1];
        b[n] = ab[n, 2];
    }

    // パラメータ再構成
    cov_chol = diag_pre_multiply(sigma_vec, corr_chol);
}

model {
    ab0[1] ~ normal(400, 200);
    ab0[2] ~ normal(15, 15);

    sigma_vec[1] ~ student_t(4, 0, 200);
    sigma_vec[2] ~ student_t(4, 0, 20);

    corr_chol ~ lkj_corr_cholesky(Nu);

    ab ~ multi_normal_cholesky(ab0, cov_chol);
    Y ~ normal(a[KID] + b[KID] .* X, sigma);
}

generated quantities {
    matrix[2, 2] corr;
    matrix[2, 2] cov;
    corr = multiply_lower_tri_self_transpose(corr_chol);
    cov = multiply_lower_tri_self_transpose(cov_chol);
}