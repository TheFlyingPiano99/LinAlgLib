//#include "TinyLA.h"
#include "TinyLA_ET.h"
#include <iostream>
#include <print>
#include <complex>

void print_expr(const auto& expr) {
    std::println("{} = {}", expr.to_string(), expr.eval());
}

int main() {
    /*
    // Create 3D vectors
    // Create a fixed size vector:
    auto v1 = TinyLA::Vec<double, 3>{1.0, 2.0, 3.0};
    // Alias for common sizes and types:
    auto v2 = TinyLA::DVec3{4.0, 5.0, 6.0};

    // Vector arithmetic:
    auto sum = v1 + v2;           // Vector addition
    auto scaled = v1 * 2.0;       // Scalar multiplication
    auto dotProduct = TinyLA::dot(v1, v2);  // Dot product
    auto crossProduct = TinyLA::cross(v1, v2);  // Cross product
    auto twoNorm = TinyLA::norm<2>(v1);
    std::cout << TinyLA::to_string(twoNorm) << std::endl;

    // Create a 4x4 identity matrix
    auto matA = TinyLA::Mat<double, 4, 4>::identity();
    auto matB = TinyLA::DMat4::identity();
    matA[0][1] = 2.0;  // Modify an element

    // Matrix multiplication
    auto matC = matA * matB + TinyLA::transpose(matA);

    // Matrix-vector multiplication
    auto v = TinyLA::Vec4<double>{1.0, 2.0, 3.0, 4.0};
    auto result = matC * v;

    std::cout << "Resulting vector: " << TinyLA::to_string(result) << std::endl;

    // Create a dual number for automatic differentiation
    auto x = TinyLA::Dual<double>{TinyLA::PI<double>, 1.0};  // f(x) = π, df/dx = 1

    // Compute function and derivative simultaneously
    auto result2 = TinyLA::sin(x) + 1.0;

    std::cout << "f(x) = sin(pi) + 1 = " << result2.fx() << std::endl;
    std::cout << "f'(x) = cos(pi) = " << result2.dfxdx() << std::endl;

    
    // Vectors with complex dual numbers
    auto cVec = TinyLA::Vec3<TinyLA::Dual<std::complex<double>>>{};
    cVec.x() = TinyLA::initVariableWithDiffOrder<1>(std::complex<double>{1.0, 0.0});
    cVec.y() = TinyLA::initVariableWithDiffOrder<1>(std::complex<double>{0.5, 2.0});
    cVec.z() = TinyLA::initVariableWithDiffOrder<1>(std::complex<double>{1.0, 1.0});

    auto complexResult = TinyLA::dot(cVec, cVec);   // Sum_i a_i * conj(b_i)
    std::cout << "Complex dot product: " << TinyLA::to_string(complexResult) << std::endl;


    // Solve quadratic equation: x² + 1 = 0
    auto a = 1.0;
    auto b = 2.0;
    auto c = 4.0;
    auto solution = TinyLA::solveQuadraticEquation<TinyLA::RootDomain::Complex>(a, b, c);

    std::cout << "Number of roots: " << solution.root_count << std::endl;
    std::cout << "Root 1: " << TinyLA::to_string(std::get<0>(solution.roots)) << std::endl;
    std::cout << "Root 2: " << TinyLA::to_string(std::get<1>(solution.roots)) << std::endl;
    */

    TinyLA::Scalar<double, 0> x0 = 0.5;
    TinyLA::Scalar<double, 1> x1 = 6.0;
    TinyLA::Scalar<double, 2> s2 = 7.0;
    
    print_expr(x0);
    print_expr(x1);
    print_expr(s2);

    // Simplified expression to avoid compiler crash
    auto s = (x0 * x1).derivate<0>();
    print_expr(s);
        

    return 0;
}
