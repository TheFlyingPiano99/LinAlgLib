//#include "TinyLA.h"
#include "TinyLA_ET.h"
#include <iostream>
#include <print>
#include <complex>

void print_expr(const auto& expr) {
    std::println("{} = {}", expr.to_string(), expr.eval());
}

int main() {
    TinyLA::Scalar<double, 0> s0 = 0.5;
    TinyLA::Scalar<double, 1> s1 = 6.0;
    TinyLA::Scalar<double, 2> s2 = 7.0;
    
    print_expr(s0);
    print_expr(s1);
    print_expr(s2);

    auto s_res = log(TinyLA::Euler<double>);
    print_expr(s_res);

    return 0;
}
