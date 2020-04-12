data {
    int N;
    vector[N] X;
    vector[N] Y;
    int N_group;
    int<lower=1, upper=N_group> KID[N];
}

parameters {
    vector[2] ab0;
    vector[2] ab[N_group];
    real<lower=0> sigma;

    real<lower=0> s_a;
    real<lower=0> s_b;

    // パラメータ再設定
    real<lower=0> rho;
}

transformed parameters {
    vector[N_group] a;
    vector[N_group] b;

    matrix[2, 2] cov_ab;

    for (n in 1:N_group) {
        a[n] = ab[n, 1];
        b[n] = ab[n, 2];
    }

    // パラメータ再構成
    cov_ab[1, 1] = square(s_a);
    cov_ab[1, 2] = s_a * s_b * rho;
    cov_ab[2, 1] = s_a * s_b * rho;
    cov_ab[2, 2] = square(s_b);
}

model {
    ab0[1] ~ normal(400, 200);
    ab0[2] ~ normal(15, 15);

    s_a ~ student_t(4, 0, 200);
    s_b ~ student_t(4, 0, 20);

    ab ~ multi_normal(ab0, cov_ab);
    Y ~ normal(a[KID] + b[KID] .* X, sigma);
}