
#include "linalg.h"
#include <print>


int main() {

    constexpr auto a = linalg::Dual<double>{3.14159, 1.0};
    const auto b = linalg::sin(a);
    auto result = b;
    std::print("Result: fx = {}, dfxdx = {}\n", result.fx(), result.dfxdx());

    return 0;
}

