data {
    int N;
    vector[N] x;
    real a;
    real b;
}

transformed parameters {
    real x_min;
    real a_b_min;

    x_min = min(x);
    a_b_min = fmin(a, b);

    print("x_min=", x_min, "a_b_min=", a_b_min)
}