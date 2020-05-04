data {
    int N;
    int N_pred;
    int L;
    vector[N] Y;
}

parameters {
    vector[N] mu;
    vector[N] season;
    real<lower=0> sigma_mu;
    real<lower=0> sigma_season;
    real<lower=0> sigma_y;
}

transformed parameters {
     vector[N] y_mean;
     y_mean = mu + season;
}

model {
    for (n in 2:N) {
        mu[n] ~ normal(mu[n-1], sigma_mu);
    }
    for (n in L:N) {
        season[n] ~ normal(-sum(season[(n-(L-1)):(n-1)]), sigma_season);
        Y[n] ~ normal(y_mean[n], sigma_y);
    }
}

generated quantities {
    vector[N+N_pred] mu_pred;
    vector[N+N_pred] season_pred;
    vector[N+N_pred] y_mean_pred;
    vector[N_pred] y_pred;

    mu_pred[1:N] = mu;
    season_pred[1:N] = season;
    y_mean_pred[1:N] = mu+season;
    for (n in 1:N_pred) {
        mu_pred[N+n] = normal_rng(mu_pred[N+n-1], sigma_mu);
        season_pred[N+n] = normal_rng(-sum(season_pred[(N+n-(L-1)):(N+n-1)]), sigma_season);
        y_mean_pred[N+n] = mu_pred[N+n]+season_pred[N+n];
        y_pred[n] = normal_rng(y_mean_pred[n], sigma_y);
    }
}
