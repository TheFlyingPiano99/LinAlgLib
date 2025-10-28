
#include "linalg.h"
#include <print>
#include <string>
#include <typeinfo>


int main() {

    constexpr const auto x = linalg::Dual<double>{linalg::PI<double>, 1.0};
    auto result = linalg::sin(x) + 1.0;
    std::println("Result: fx = {}, dfxdx = {}", result.fx(), result.dfxdx());

    auto cVec = linalg::Vec3<linalg::Dual<cuda::std::complex<float>>>{};
    cVec.x() = linalg::Dual<cuda::std::complex<float>>{1.0, 1.0};
    cVec.y() = linalg::Dual<cuda::std::complex<float>>{1.0, 1.0};
    cVec.z() = linalg::Dual<cuda::std::complex<float>>{1.0, 1.0};
    auto cResult = linalg::dot(cVec, cVec);
    std::println("Complex Result: fx = {}, dfxdx = {}", cResult.fx().real(), cResult.dfxdx().real());

    constexpr auto matA = linalg::Mat<double, 4, 4>::identity();
    constexpr auto matB = linalg::Mat<double, 4, 4>::identity();
    constexpr auto matC = matA * matB + linalg::transpose(matA);
    constexpr auto matCAdj = linalg::adj(matC);
    std::print("Matrix C: \n");
    for (uint32_t i = 0; i < 4; ++i) {
        for (uint32_t j = 0; j < 4; ++j) {
            std::print("{} ", matC[i][j]);
        }
        std::print("\n");
    }

    auto v = linalg::Vec4<std::complex<double>>{2.0, 2.0, 2.0, 2.0};
    auto resVec = matC * v;
    std::println("Resulting Vector: {}", linalg::to_string(resVec));

    auto scalarRes = v * linalg::Mat<double, 4, 4>::identity() * v;
    std::println("v * M * v = {}", linalg::to_string(scalarRes));

    std::pair<double, double> roots = {0.0, 0.0};
    auto solution = linalg::solveQuadraticEquation<linalg::RootDomain::Complex>(1.0, 0.0, 1.0);
    std::print("Type of roots: {}\n", typeid(std::get<0>(solution.roots)).name());
    std::print("Number of roots: {}\n", solution.root_count);
    std::print("Roots: {}, {}\n", linalg::to_string(std::get<0>(solution.roots)), linalg::to_string(std::get<1>(solution.roots)));

    return 0;
}

