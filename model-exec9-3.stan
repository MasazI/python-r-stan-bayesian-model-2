data {
    int N;
    vector[N] a;
    vector[N] b;

    row_vector[N] ra;
}

transformed parameters {
    real ip;
    real ip_op;

    ip = dot_product(a, b);
    print("ip=", ip)

    ip_op = ra * b;
    print("ip_op=", ip_op)
}