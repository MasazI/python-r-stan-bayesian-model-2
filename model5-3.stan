data {
    int N;
    real<lower=0, upper=1> score[N];
    int<lower=0, upper=1> a[N];
    real<lower=0, upper=1> y[N];
}

parameters {
    real Intercept;
    real b_s;
    real b_a;
    real<lower=0> sigma;
}

transformed parameters {
    vector[N] mu;
    for (i in 1:N) {
        mu[i] = Intercept + b_s * score[i] +  b_a * a[i];
    }
}

model {
    for (i in 1:N) {
        y[i] ~ normal(mu[i], sigma);
    }
}

generated quantities {
    real y_pred[N];
    real noise[N];
    for (i in 1:N) {
        y_pred[i] = normal_rng(mu[i], sigma);

        # 練習問題5(2)
        noise[i] = y[i] - mu[i];
    }
}