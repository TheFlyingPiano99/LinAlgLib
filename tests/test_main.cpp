
#include "linalg.h"
#include <print>
#include <string>
#include <typeinfo>


int main() {

    constexpr const auto x = linalg::Dual<double>{linalg::PI<double>, 1.0};
    auto result = linalg::sin(x) + 1.0;
    std::println("Result: {}", linalg::to_string(result));

    auto cVec = linalg::Vec3<linalg::Dual<cuda::std::complex<float>>>{};
    cVec.x() = linalg::Dual<cuda::std::complex<float>>{1.0, 1.0};
    cVec.y() = linalg::Dual<cuda::std::complex<float>>{1.0, 1.0};
    cVec.z() = linalg::Dual<cuda::std::complex<float>>{1.0, 1.0};
    auto cResult = linalg::dot(cVec, cVec);
    std::println("cVec: {}", linalg::to_string(cVec));
    std::println("Complex Result: {}", linalg::to_string(cResult));

    constexpr auto matA = linalg::Mat<double, 4, 4>::identity();
    constexpr auto matB = linalg::Mat<double, 4, 4>::identity();
    constexpr auto matC = matA * matB + linalg::transpose(matA);
    constexpr auto matCAdj = linalg::adj(matC);
    std::println("Matrix C:\n{}", linalg::to_string(matCAdj));

    auto v = linalg::Vec4<std::complex<double>>{2.0, 2.0, 2.0, 2.0};
    auto resVec = matC * v;
    std::println("Resulting Vector: {}", linalg::to_string(resVec));

    auto scalarRes = v * linalg::Mat<double, 4, 4>::identity() * v;
    std::println("v * M * v = {}", linalg::to_string(scalarRes));

    auto solution = linalg::solveQuadraticEquation<linalg::RootDomain::Complex>(1.0, 0.0, 1.0);
    std::print("Type of roots: {}\n", typeid(std::get<0>(solution.roots)).name());
    std::print("Number of roots: {}\n", solution.root_count);
    std::print("Roots: {}, {}\n", linalg::to_string(std::get<0>(solution.roots)), linalg::to_string(std::get<1>(solution.roots)));

    return 0;
}

