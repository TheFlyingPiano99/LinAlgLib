
#include "linalg.h"
#include <print>


int main() {

    constexpr auto x = linalg::Dual<double>{linalg::PI<double>, 1.0};
    auto result = linalg::sin(x) + 1.0;
    std::println("Result: fx = {}, dfxdx = {}", result.fx(), result.dfxdx());

    auto cVec = linalg::Vec3<linalg::Dual<cuda::std::complex<float>>>{};
    cVec.x() = linalg::Dual<cuda::std::complex<float>>{1.0, 1.0};
    cVec.y() = linalg::Dual<cuda::std::complex<float>>{1.0, 1.0};
    cVec.z() = linalg::Dual<cuda::std::complex<float>>{1.0, 1.0};
    auto cResult = linalg::dot(cVec, cVec);
    std::println("Complex Result: fx = {}, dfxdx = {}", cResult.fx().real(), cResult.dfxdx().real());

    auto matA = linalg::Mat<double, 4, 4>::identity();
    auto matB = linalg::Mat<double, 4, 4>::identity();
    auto matC = matA * matB + linalg::transpose(matA);
    std::print("Matrix C: \n");
    for (uint32_t i = 0; i < 4; ++i) {
        for (uint32_t j = 0; j < 4; ++j) {
            std::print("{} ", matC[i][j]);
        }
        std::print("\n");
    }

    auto v = linalg::Vec4<double>{2.0, 2.0, 2.0, 2.0};
    auto resVec = matC * v;
    std::println("Resulting Vector: ({}, {}, {}, {})", resVec.x(), resVec.y(), resVec.z(), resVec.w());

    auto scalarRes = v * linalg::Mat<double, 4, 4>::identity() * v;
    std::println("Scalar Result: {}", scalarRes);

    std::pair<double, double> roots = {0.0, 0.0};
    auto noOfRoots = linalg::solveQuadraticEquation(1.0, -3.0, 2.0, roots.first, roots.second);
    std::print("Number of roots: {}\n", noOfRoots);
    std::print("Roots: {}, {}\n", roots.first, roots.second);

    return 0;
}

