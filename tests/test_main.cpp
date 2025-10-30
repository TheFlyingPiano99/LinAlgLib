
#include "TinyLA.h"
#include <print>
#include <string>
#include <typeinfo>


int main() {

    constexpr const auto x = TinyLA::Dual<double>{TinyLA::PI<double>, 1.0};
    auto result = TinyLA::sin(x) + 1.0;
    std::println("Result: {}", TinyLA::to_string(result));

    auto cVec = TinyLA::Vec3<TinyLA::Dual<cuda::std::complex<float>>>{};
    cVec.x() = TinyLA::Dual<cuda::std::complex<float>>{1.0, 1.0};
    cVec.y() = TinyLA::Dual<cuda::std::complex<float>>{1.0, 1.0};
    cVec.z() = TinyLA::Dual<cuda::std::complex<float>>{1.0, 1.0};
    auto cResult = TinyLA::dot(cVec, cVec);
    std::println("cVec: {}", TinyLA::to_string(cVec));
    std::println("Complex Result: {}", TinyLA::to_string(cResult));

    auto matA = TinyLA::Mat<double, 4, 4>::identity();
    matA[0][1] = 2.0;
    constexpr auto matB = TinyLA::Mat<double, 4, 4>::identity();
    auto matC = matA * matB + TinyLA::transpose(matA);
    auto matD = TinyLA::Mat<double, 4, 3>{
        {1.0, 0.0, 0.0},
        {0.0, 5.0, 0.0},
        {0.0, 0.0, 2.0},
        {0.0, 0.0, 1.0}
    };
    const auto tensorProd = TinyLA::kronecker(matA, matD);
    std::println("Tensor Product:\n{}", TinyLA::to_string(tensorProd));
    std::println("Matrix C:\n{}", TinyLA::to_string(matC));
    auto matCAdj = TinyLA::adj(matC);
    std::println("Matrix C:\n{}", TinyLA::to_string(matCAdj));

    auto v = TinyLA::Vec4<std::complex<double>>{2.0, 2.0, 2.0, 2.0};
    auto resVec = matC * v;
    std::println("Resulting Vector: {}", TinyLA::to_string(resVec));

    auto scalarRes = v * TinyLA::Mat<double, 4, 4>::identity() * v;
    std::println("v * M * v = {}", TinyLA::to_string(scalarRes));

    auto solution = TinyLA::solveQuadraticEquation<TinyLA::RootDomain::Complex>(1.0, 0.0, 1.0);
    std::print("Type of roots: {}\n", typeid(std::get<0>(solution.roots)).name());
    std::print("Number of roots: {}\n", solution.root_count);
    std::print("Roots: {}, {}\n", TinyLA::to_string(std::get<0>(solution.roots)), TinyLA::to_string(std::get<1>(solution.roots)));


    std::println("v = {}", TinyLA::to_string(v));


    std::println("All tests passed!");
    return 0;
}

