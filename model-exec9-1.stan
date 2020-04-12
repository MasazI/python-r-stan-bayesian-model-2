data {
    int N;
    vector<lower=0, upper=1>[N] score;
    vector<lower=0, upper=1>[N] a;
    vector[N] y;
}

parameters {
    real Intercept;
    real b_s;
    real b_a;
    real<lower=0> sigma;
}

transformed parameters {
    vector[N] mu;
    mu = Intercept + b_s * score +  b_a * a;
}

model {
    y ~ normal(mu, sigma);
}

generated quantities {
    real y_pred[N];
    real noise[N];
    for (i in 1:N) {
        y_pred[i] = normal_rng(mu[i], sigma);

        // 練習問題5(2)
        noise[i] = y[i] - mu[i];
    }
}