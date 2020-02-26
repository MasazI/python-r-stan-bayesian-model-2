data {
    int N; // Person
    int C; // コース
    int I; // 授業

    real<lower=0, upper=1> Score[N];
    int<lower=0, upper=1> A[N];

    int<lower=1, upper=N> PID[I];
    int<lower=1, upper=C> CID[I];
    real<lower=0, upper=1> W[I];
    int<lower=0, upper=1> Y[I];
}

parameters {
    real b1; // 授業ごと
    real b2; // アルバイトの好き嫌い
    real b3; // スコア
    real b4; // 天気
    real bp[N]; // 学生ごと
    real bc[C]; // コースごと
    real<lower=0> sigma_p; // 人ごとのばらつき
    real<lower=0> sigma_c; // コースごとのばらつき
}

transformed parameters {
    real x[I];
    real xw[I];
    real xp[N];
    real xc[C];
    real q[I];
    for (n in 1:N) {
        xp[n] = b2*A[n] + b3*Score[n] + bp[n];
    }
    for (c in 1:C) {
        xc[c] = bc[c];
    }
    for (i in 1:I) {
        xw[i] = b4 * W[i];
        x[i] = b1 + xp[PID[i]] + xc[CID[i]] + xw[i];
        q[i] = inv_logit(x[i]);
    }
}

model {
    for (n in 1:N) {
        bp[n] ~ normal(0, sigma_p);
    }
    for (c in 1:C) {
        bc[c] ~ normal(0, sigma_c);
    }
    for (i in 1:I) {
        Y[i] ~ bernoulli(q[i]);
    }
}

