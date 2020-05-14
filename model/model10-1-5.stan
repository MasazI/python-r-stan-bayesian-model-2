data {
    int N; // num of games
    int G; // num of groups(usagi, kame)
    int LW[N, 2];
}

parameters {
    real b;
    ordered[2] performance[N];
}

transformed parameters {
    real mu[G];
    mu[1] = 0;
    mu[2] = b;
}

model {
    for (n in 1:N) {
        for (j in 1:2) {
            performance[n, j] ~ normal(mu[LW[n, j]], 1);
        }
    }
}

