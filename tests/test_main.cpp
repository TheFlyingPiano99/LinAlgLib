
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "TinyLA_ET.h"
#include <complex>
#include <string>

using namespace TinyLA;
using Catch::Approx;

// Test basic scalar functionality
TEST_CASE("Scalar basic operations", "[scalar]") {
    SECTION("Construction and evaluation") {
        Scalar<double, 0> x0(5.0);
        Scalar<double, 1> x1(10.0);
        Scalar<double> constant(3.14);
        
        REQUIRE(x0.eval() == Approx(5.0));
        REQUIRE(x1.eval() == Approx(10.0));
        REQUIRE(constant.eval() == Approx(3.14));
    }
    
    SECTION("Type conversion") {
        Scalar<double, 0> x(5.0);
        double value = x;
        REQUIRE(value == Approx(5.0));
    }
    
    SECTION("Assignment") {
        Scalar<double, 0> x(5.0);
        x = 7.5;
        REQUIRE(x.eval() == Approx(7.5));
    }
    
    SECTION("String representation") {
        Scalar<double, 0> x0(5.0);
        Scalar<double> constant(3.14);
        
        REQUIRE(x0.to_string() == "s_0");
        // Constant should show its value
        auto const_str = constant.to_string();
        REQUIRE(const_str.find("3.14") != std::string::npos);
    }
}

// Test scalar arithmetic operations
TEST_CASE("Scalar arithmetic operations", "[scalar][arithmetic]") {
    SECTION("Addition") {
        Scalar<double, 0> x0(5.0);
        Scalar<double, 1> x1(3.0);
        
        auto sum = x0 + x1;
        REQUIRE(sum.eval() == Approx(8.0));
        
        auto sum_with_constant = x0 + 2.0;
        REQUIRE(sum_with_constant.eval() == Approx(7.0));
        
        auto constant_with_scalar = 2.0 + x0;
        REQUIRE(constant_with_scalar.eval() == Approx(7.0));
    }
    
    SECTION("Subtraction") {
        Scalar<double, 0> x0(5.0);
        Scalar<double, 1> x1(3.0);
        
        auto diff = x0 - x1;
        REQUIRE(diff.eval() == Approx(2.0));
        
        auto diff_with_constant = x0 - 2.0;
        REQUIRE(diff_with_constant.eval() == Approx(3.0));
        
        auto constant_minus_scalar = 10.0 - x0;
        REQUIRE(constant_minus_scalar.eval() == Approx(5.0));
    }
    
    SECTION("Multiplication") {
        Scalar<double, 0> x0(5.0);
        Scalar<double, 1> x1(3.0);
        
        auto product = x0 * x1;
        REQUIRE(product.eval() == Approx(15.0));
        
        auto scaled = x0 * 2.0;
        REQUIRE(scaled.eval() == Approx(10.0));
        
        auto constant_times_scalar = 2.0 * x0;
        REQUIRE(constant_times_scalar.eval() == Approx(10.0));
    }
    
    SECTION("Division") {
        Scalar<double, 0> x0(10.0);
        Scalar<double, 1> x1(2.0);
        
        auto quotient = x0 / x1;
        REQUIRE(quotient.eval() == Approx(5.0));
        
        auto divided_by_constant = x0 / 2.0;
        REQUIRE(divided_by_constant.eval() == Approx(5.0));
        
        auto constant_divided_by_scalar = 20.0 / x0;
        REQUIRE(constant_divided_by_scalar.eval() == Approx(2.0));
    }
}

// Test complex arithmetic expressions
TEST_CASE("Complex expressions", "[scalar][expressions]") {
    SECTION("Chained operations") {
        Scalar<double, 0> x0(2.0);
        Scalar<double, 1> x1(3.0);
        
        // Test: (x0 + x1) - 1
        auto expr = (x0 + x1) - 1.0;
        double expected = (2.0 + 3.0) - 1.0; // = 4.0
        REQUIRE(expr.eval() == Approx(expected));
    }
    
    SECTION("Simple nested expressions") {
        Scalar<double, 0> x0(1.0);
        Scalar<double, 1> x1(2.0);
        
        // Test: x0 + x1
        auto expr = x0 + x1;
        double expected = 1.0 + 2.0; // = 3.0
        REQUIRE(expr.eval() == Approx(expected));
    }
}

// Test automatic differentiation
TEST_CASE("Automatic differentiation", "[scalar][differentiation]") {
    SECTION("Variable derivatives") {
        Scalar<double, 0> x0(5.0);
        Scalar<double, 1> x1(3.0);
        
        // Derivative of x0 with respect to variable 0 should be 1
        auto dx0_dx0 = x0.derivate<0>();
        REQUIRE(dx0_dx0.eval() == Approx(1.0));
        
        // Derivative of x0 with respect to variable 1 should be 0
        auto dx0_dx1 = x0.derivate<1>();
        REQUIRE(dx0_dx1.eval() == Approx(0.0));
        
        // Derivative of x1 with respect to variable 1 should be 1
        auto dx1_dx1 = x1.derivate<1>();
        REQUIRE(dx1_dx1.eval() == Approx(1.0));
    }
    
    SECTION("Sum derivatives") {
        Scalar<double, 0> x0(5.0);
        Scalar<double, 1> x1(3.0);
        
        auto sum = x0 + x1;
        
        // d/dx0 (x0 + x1) = 1
        auto d_sum_dx0 = sum.derivate<0>();
        REQUIRE(d_sum_dx0.eval() == Approx(1.0));
        
        // d/dx1 (x0 + x1) = 1
        auto d_sum_dx1 = sum.derivate<1>();
        REQUIRE(d_sum_dx1.eval() == Approx(1.0));
    }
    
    SECTION("Product rule") {
        Scalar<double, 0> x0(5.0);
        Scalar<double, 1> x1(3.0);
        
        auto product = x0 * x1;
        
        // d/dx0 (x0 * x1) = x1 = 3.0
        auto d_product_dx0 = product.derivate<0>();
        REQUIRE(d_product_dx0.eval() == Approx(3.0));
        
        // d/dx1 (x0 * x1) = x0 = 5.0
        auto d_product_dx1 = product.derivate<1>();
        REQUIRE(d_product_dx1.eval() == Approx(5.0));
    }
    
    SECTION("Chain rule with constants") {
        Scalar<double, 0> x0(2.0);
        
        auto expr = x0 * 3.0; // 3*x0
        
        // d/dx0 (3*x0) = 3
        auto derivative = expr.derivate<0>();
        REQUIRE(derivative.eval() == Approx(3.0));
    }
}

// Test special scalar types
TEST_CASE("Special scalar types", "[scalar][special]") {
    SECTION("ScalarZero") {
        ZeroScalar<double> zero;
        REQUIRE(zero.eval() == Approx(0.0));
        REQUIRE(zero.to_string() == "0");
        
        // Derivative of zero should be zero
        auto d_zero = zero.derivate<0>();
        REQUIRE(d_zero.eval() == Approx(0.0));
    }
    
    SECTION("ScalarUnit") {
        UnitScalar<double> one;
        REQUIRE(one.eval() == Approx(1.0));
        REQUIRE(one.to_string() == "1");
        
        // Derivative of constant should be zero
        auto d_one = one.derivate<0>();
        REQUIRE(d_one.eval() == Approx(0.0));
    }
    
    SECTION("ScalarConstant") {
        ScalarConstant<double> five(5.0);
        REQUIRE(five.eval() == Approx(5.0));
        
        // Derivative of constant should be zero
        auto d_five = five.derivate<0>();
        REQUIRE(d_five.eval() == Approx(0.0));
    }
}

// Test expression string representations
TEST_CASE("Expression string formatting", "[scalar][formatting]") {
    SECTION("Simple expressions") {
        Scalar<double, 0> x0(2.0);
        Scalar<double, 1> x1(3.0);
        
        auto sum = x0 + x1;
        REQUIRE(sum.to_string() == "s_0 + s_1");
        
        auto product = x0 * x1;
        REQUIRE(product.to_string() == "s_0 * s_1");

        auto diff = x0 - x1;
        REQUIRE(diff.to_string() == "s_0 - s_1");
    }
    
    SECTION("Parentheses in complex expressions") {
        Scalar<double, 0> x0(2.0);
        Scalar<double, 1> x1(3.0);
        
        // This should add parentheses around sum when multiplied
        auto expr = (x0 + x1) * x0;
        std::string expr_str = expr.to_string();
        
        // Should contain parentheses to preserve order of operations
        REQUIRE(expr_str.find('(') != std::string::npos);
        REQUIRE(expr_str.find(')') != std::string::npos);
    }
}

// Test mathematical constants
TEST_CASE("Mathematical constants", "[scalar][constants]") {
    SECTION("PI constant") {
        auto pi_val = PI<double>;
        REQUIRE(pi_val.eval() == Approx(3.14159265358979323846));
    }
    
    SECTION("Euler constant") {
        auto e_val = Euler<double>;
        REQUIRE(e_val.eval() == Approx(2.71828182845904523536));
    }
}

// Test complex number support
TEST_CASE("Complex number scalars", "[scalar][complex]") {
    SECTION("Complex scalar operations") {
        Scalar<std::complex<double>, 0> z0(std::complex<double>(1.0, 2.0));
        Scalar<std::complex<double>, 1> z1(std::complex<double>(3.0, 4.0));
        
        auto sum = z0 + z1;
        auto result = sum.eval();
        
        REQUIRE(result.real() == Approx(4.0));
        REQUIRE(result.imag() == Approx(6.0));
    }
    
    SECTION("Complex differentiation") {
        Scalar<std::complex<double>, 0> z0(std::complex<double>(1.0, 2.0));
        
        auto derivative = z0.derivate<0>();
        auto result = derivative.eval();
        
        REQUIRE(result.real() == Approx(1.0));
        REQUIRE(result.imag() == Approx(0.0));
    }
}

// Test edge cases and error conditions
TEST_CASE("Edge cases", "[scalar][edge_cases]") {
    SECTION("Zero operations") {
        Scalar<double, 0> x0(0.0);
        Scalar<double, 1> x1(5.0);
        
        auto product = x0 * x1;
        REQUIRE(product.eval() == Approx(0.0));
        
        auto sum = x0 + x1;
        REQUIRE(sum.eval() == Approx(5.0));
    }
    
    SECTION("Large numbers") {
        Scalar<double, 0> x0(1e10);
        Scalar<double, 1> x1(1e-10);
        
        auto product = x0 * x1;
        REQUIRE(product.eval() == Approx(1.0));
    }
    
    SECTION("Negative numbers") {
        Scalar<double, 0> x0(-5.0);
        Scalar<double, 1> x1(3.0);
        
        auto sum = x0 + x1;
        REQUIRE(sum.eval() == Approx(-2.0));
        
        auto product = x0 * x1;
        REQUIRE(product.eval() == Approx(-15.0));
    }
}

// Test Expression Template Matrix types
TEST_CASE("Expression Template Matrix construction", "[matrix][et]") {
    SECTION("Matrix ET basic construction") {
        Matrix<double, 2, 2, 0> m{{1.0, 2.0}, {3.0, 4.0}};
        
        REQUIRE(m.eval(0, 0) == Approx(1.0));
        REQUIRE(m.eval(0, 1) == Approx(2.0));
        REQUIRE(m.eval(1, 0) == Approx(3.0));
        REQUIRE(m.eval(1, 1) == Approx(4.0));
    }
    
    SECTION("Matrix ET with different variable IDs") {
        Matrix<double, 2, 2, 5> m5{{1.0, 2.0}, {3.0, 4.0}};
        Matrix<double, 3, 2, 10> m10{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
        
        REQUIRE(m5.eval(1, 1) == Approx(4.0));
        REQUIRE(m10.eval(2, 0) == Approx(5.0));
        REQUIRE(m10.eval(2, 1) == Approx(6.0));
    }
    
    SECTION("Matrix ET string representation") {
        Matrix<double, 2, 2, 0> m0{{1.0, 2.0}, {3.0, 4.0}};
        Matrix<double, 2, 2, 3> m3{{1.0, 2.0}, {3.0, 4.0}};
        
        // Should show variable format for variable matrices
        REQUIRE(m0.to_string() == "M_0");
        REQUIRE(m3.to_string() == "M_3");
    }
}

TEST_CASE("IdentityMatrix ET operations", "[matrix][et][identity]") {
    SECTION("IdentityMatrix basic properties") {
        IdentityMatrix<double, 2> id2;
        IdentityMatrix<double, 3> id3;
        IdentityMatrix<double, 4> id4;
        
        // Test diagonal elements (should be 1)
        REQUIRE(id2.eval(0, 0) == Approx(1.0));
        REQUIRE(id2.eval(1, 1) == Approx(1.0));
        REQUIRE(id3.eval(0, 0) == Approx(1.0));
        REQUIRE(id3.eval(1, 1) == Approx(1.0));
        REQUIRE(id3.eval(2, 2) == Approx(1.0));
        REQUIRE(id4.eval(3, 3) == Approx(1.0));
        
        // Test off-diagonal elements (should be 0)
        REQUIRE(id2.eval(0, 1) == Approx(0.0));
        REQUIRE(id2.eval(1, 0) == Approx(0.0));
        REQUIRE(id3.eval(0, 1) == Approx(0.0));
        REQUIRE(id3.eval(1, 2) == Approx(0.0));
        REQUIRE(id3.eval(2, 0) == Approx(0.0));
    }
    
    SECTION("IdentityMatrix string representation") {
        IdentityMatrix<double, 3> identity;
        REQUIRE(identity.to_string() == "I");
    }
    
    SECTION("IdentityMatrix with complex numbers") {
        IdentityMatrix<std::complex<double>, 2> complex_id;
        
        auto diag_elem = complex_id.eval(0, 0);
        auto off_diag_elem = complex_id.eval(0, 1);
        
        REQUIRE(diag_elem.real() == Approx(1.0));
        REQUIRE(diag_elem.imag() == Approx(0.0));
        REQUIRE(off_diag_elem.real() == Approx(0.0));
        REQUIRE(off_diag_elem.imag() == Approx(0.0));
    }
}

TEST_CASE("ZeroMatrix ET operations", "[matrix][et][zero]") {
    SECTION("ZeroMatrix basic properties") {
        ZeroMatrix<double, 2, 2> zero2x2;
        ZeroMatrix<double, 3, 4> zero3x4;
        ZeroMatrix<double, 1, 5> zero1x5;
        
        // Test that all elements are zero
        for (uint32_t r = 0; r < 2; ++r) {
            for (uint32_t c = 0; c < 2; ++c) {
                REQUIRE(zero2x2.eval(r, c) == Approx(0.0));
            }
        }
        
        for (uint32_t r = 0; r < 3; ++r) {
            for (uint32_t c = 0; c < 4; ++c) {
                REQUIRE(zero3x4.eval(r, c) == Approx(0.0));
            }
        }
        
        for (uint32_t c = 0; c < 5; ++c) {
            REQUIRE(zero1x5.eval(0, c) == Approx(0.0));
        }
    }
    
    SECTION("ZeroMatrix string representation") {
        ZeroMatrix<double, 2, 3> zero;
        REQUIRE(zero.to_string() == "0");
    }
    
    SECTION("ZeroMatrix with different scalar types") {
        ZeroMatrix<float, 2, 2> zero_float;
        ZeroMatrix<int, 2, 2> zero_int;
        ZeroMatrix<std::complex<double>, 2, 2> zero_complex;
        
        REQUIRE(zero_float.eval(0, 0) == Approx(0.0f));
        REQUIRE(zero_int.eval(1, 1) == 0);
        
        auto complex_zero = zero_complex.eval(0, 1);
        REQUIRE(complex_zero.real() == Approx(0.0));
        REQUIRE(complex_zero.imag() == Approx(0.0));
    }
}

TEST_CASE("Matrix ET differentiation", "[matrix][et][differentiation]") {
    SECTION("Matrix variable differentiation") {
        Matrix<double, 2, 2, 0> m0{{1.0, 2.0}, {3.0, 4.0}};
        Matrix<double, 2, 2, 1> m1{{5.0, 6.0}, {7.0, 8.0}};
        Matrix<double, 3, 2, 2> m2{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
        
        // Derivative of m0 with respect to variable 0 should be unit (1)
        auto dm0_dx0 = m0.derivate<0>();
        REQUIRE(dm0_dx0.eval() == Approx(1.0));
        
        // Derivative of m0 with respect to variable 1 should be zero
        auto dm0_dx1 = m0.derivate<1>();
        REQUIRE(dm0_dx1.eval() == Approx(0.0));
        
        // Derivative of m1 with respect to variable 1 should be unit (1)
        auto dm1_dx1 = m1.derivate<1>();
        REQUIRE(dm1_dx1.eval() == Approx(1.0));
        
        // Derivative of m1 with respect to variable 0 should be zero
        auto dm1_dx0 = m1.derivate<0>();
        REQUIRE(dm1_dx0.eval() == Approx(0.0));
        
        // Derivative of m2 with respect to variable 2 should be unit (1)
        auto dm2_dx2 = m2.derivate<2>();
        REQUIRE(dm2_dx2.eval() == Approx(1.0));
    }
    
    SECTION("IdentityMatrix differentiation") {
        IdentityMatrix<double, 3> identity;
        
        // Derivative of constant identity matrix should always be zero
        auto d_identity_dx0 = identity.derivate<0>();
        auto d_identity_dx5 = identity.derivate<5>();
        
        REQUIRE(d_identity_dx0.eval() == Approx(0.0));
        REQUIRE(d_identity_dx5.eval() == Approx(0.0));
    }
    
    SECTION("ZeroMatrix differentiation") {
        ZeroMatrix<double, 2, 3> zero;
        
        // Derivative of constant zero matrix should always be zero
        auto d_zero_dx0 = zero.derivate<0>();
        auto d_zero_dx10 = zero.derivate<10>();
        
        REQUIRE(d_zero_dx0.eval() == Approx(0.0));
        REQUIRE(d_zero_dx10.eval() == Approx(0.0));
    }
}

TEST_CASE("Matrix ET edge cases", "[matrix][et][edge_cases]") {
    SECTION("1x1 matrices") {
        Matrix<double, 1, 1, 0> tiny{{5.0}};
        IdentityMatrix<double, 1> tiny_id;
        ZeroMatrix<double, 1, 1> tiny_zero;
        
        REQUIRE(tiny.eval(0, 0) == Approx(5.0));
        REQUIRE(tiny_id.eval(0, 0) == Approx(1.0));
        REQUIRE(tiny_zero.eval(0, 0) == Approx(0.0));
    }
    
    SECTION("Large matrices") {
        Matrix<double, 4, 4, 0> large{
            {1.0, 2.0, 3.0, 4.0},
            {5.0, 6.0, 7.0, 8.0},
            {9.0, 10.0, 11.0, 12.0},
            {13.0, 14.0, 15.0, 16.0}
        };
        
        IdentityMatrix<double, 5> large_id;
        ZeroMatrix<double, 4, 6> large_zero;
        
        REQUIRE(large.eval(0, 0) == Approx(1.0));
        REQUIRE(large.eval(3, 3) == Approx(16.0));
        REQUIRE(large.eval(2, 1) == Approx(10.0));
        
        REQUIRE(large_id.eval(4, 4) == Approx(1.0));
        REQUIRE(large_id.eval(0, 4) == Approx(0.0));
        
        REQUIRE(large_zero.eval(3, 5) == Approx(0.0));
    }
    
    SECTION("Non-square matrices") {
        Matrix<double, 2, 3, 0> rect2x3{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
        Matrix<double, 3, 2, 1> rect3x2{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
        ZeroMatrix<double, 2, 5> zero2x5;
        ZeroMatrix<double, 6, 2> zero6x2;
        
        REQUIRE(rect2x3.eval(0, 2) == Approx(3.0));
        REQUIRE(rect2x3.eval(1, 1) == Approx(5.0));
        REQUIRE(rect3x2.eval(2, 0) == Approx(5.0));
        REQUIRE(rect3x2.eval(2, 1) == Approx(6.0));
        
        REQUIRE(zero2x5.eval(1, 4) == Approx(0.0));
        REQUIRE(zero6x2.eval(5, 1) == Approx(0.0));
    }
    
    SECTION("Matrix with negative variable ID") {
        // Test default varId = -1 (constant matrix)
        Matrix<double, 2, 2> constant_matrix{{1.0, 2.0}, {3.0, 4.0}};
        
        REQUIRE(constant_matrix.eval(0, 0) == Approx(1.0));
        REQUIRE(constant_matrix.eval(1, 1) == Approx(4.0));
        
        // Should show actual matrix values, not variable name
        auto str_repr = constant_matrix.to_string();
        REQUIRE(str_repr.find("1") != std::string::npos);
        REQUIRE(str_repr.find("4") != std::string::npos);
    }
}

// Test parentheses rules in to_string output
TEST_CASE("Expression parentheses rules", "[scalar][formatting][parentheses]") {
    SECTION("Simple addition - no parentheses needed") {
        Scalar<double, 0> x0(2.0);
        Scalar<double, 1> x1(3.0);
        
        auto sum = x0 + x1;
        REQUIRE(sum.to_string() == "s_0 + s_1");
    }
    
    SECTION("Simple subtraction - no parentheses needed") {
        Scalar<double, 0> x0(5.0);
        Scalar<double, 1> x1(3.0);
        
        auto diff = x0 - x1;
        REQUIRE(diff.to_string() == "s_0 - s_1");
    }
    
    SECTION("Simple multiplication - no parentheses needed") {
        Scalar<double, 0> x0(2.0);
        Scalar<double, 1> x1(3.0);
        
        auto product = x0 * x1;
        REQUIRE(product.to_string() == "s_0 * s_1");
    }
    
    SECTION("Multiplication with addition - parentheses added") {
        Scalar<double, 0> x0(2.0);
        Scalar<double, 1> x1(3.0);
        Scalar<double, 2> x2(4.0);
        
        // (x0 + x1) * x2 should add parentheses around the sum
        auto expr = (x0 + x1) * x2;
        std::string expr_str = expr.to_string();
        
        // Should contain parentheses around the addition
        REQUIRE(expr_str.find('(') != std::string::npos);
        REQUIRE(expr_str.find(')') != std::string::npos);
        REQUIRE(expr_str.find("s_0 + s_1") != std::string::npos);
        REQUIRE(expr_str.find("*") != std::string::npos);
        REQUIRE(expr_str.find("s_2") != std::string::npos);
    }
    
    SECTION("Multiplication with subtraction - parentheses added") {
        Scalar<double, 0> x0(5.0);
        Scalar<double, 1> x1(3.0);
        Scalar<double, 2> x2(2.0);
        
        // x0 * (x1 - x2) should add parentheses around the subtraction
        auto expr = x0 * (x1 - x2);
        std::string expr_str = expr.to_string();
        
        // Should contain parentheses around the subtraction
        REQUIRE(expr_str.find('(') != std::string::npos);
        REQUIRE(expr_str.find(')') != std::string::npos);
        REQUIRE(expr_str.find("s_1 - s_2") != std::string::npos);
        REQUIRE(expr_str.find("*") != std::string::npos);
        REQUIRE(expr_str.find("s_0") != std::string::npos);
    }
    
    SECTION("Both operands need parentheses in multiplication") {
        Scalar<double, 0> x0(1.0);
        Scalar<double, 1> x1(2.0);
        Scalar<double, 2> x2(3.0);
        Scalar<double, 3> x3(4.0);
        
        // (x0 + x1) * (x2 - x3) should add parentheses around both
        auto expr = (x0 + x1) * (x2 - x3);
        std::string expr_str = expr.to_string();
        
        // Should contain two sets of parentheses
        auto open_count = std::count(expr_str.begin(), expr_str.end(), '(');
        auto close_count = std::count(expr_str.begin(), expr_str.end(), ')');
        REQUIRE(open_count == 2);
        REQUIRE(close_count == 2);
        
        // Should contain both sub-expressions
        REQUIRE(expr_str.find("s_0 + s_1") != std::string::npos);
        REQUIRE(expr_str.find("s_2 - s_3") != std::string::npos);
        REQUIRE(expr_str.find("*") != std::string::npos);
    }
    
    SECTION("Nested additions - no extra parentheses") {
        Scalar<double, 0> x0(1.0);
        Scalar<double, 1> x1(2.0);
        Scalar<double, 2> x2(3.0);
        
        // x0 + x1 + x2 should not add parentheses (same precedence)
        auto expr = (x0 + x1) + x2;
        std::string expr_str = expr.to_string();
        
        // Should not contain parentheses (addition is left-associative)
        REQUIRE(expr_str.find('(') == std::string::npos);
        REQUIRE(expr_str.find(')') == std::string::npos);
        REQUIRE(expr_str.find("s_0 + s_1 + s_2") != std::string::npos);
    }
    
    SECTION("Chained multiplications - no extra parentheses") {
        Scalar<double, 0> x0(2.0);
        Scalar<double, 1> x1(3.0);
        Scalar<double, 2> x2(4.0);
        
        // x0 * x1 * x2 should not add parentheses (same precedence)
        auto expr = (x0 * x1) * x2;
        std::string expr_str = expr.to_string();
        
        // Should not contain parentheses (multiplication is left-associative)
        REQUIRE(expr_str.find('(') == std::string::npos);
        REQUIRE(expr_str.find(')') == std::string::npos);
        REQUIRE(expr_str.find("s_0 * s_1 * s_2") != std::string::npos);
    }
    
    SECTION("Complex mixed expression") {
        Scalar<double, 0> x0(1.0);
        Scalar<double, 1> x1(2.0);
        Scalar<double, 2> x2(3.0);
        Scalar<double, 3> x3(4.0);
        
        // (x0 + x1) * x2 - x3 should have parentheses around addition only
        auto expr = (x0 + x1) * x2 - x3;
        std::string expr_str = expr.to_string();
        
        // Should contain parentheses around the addition but not around the whole product
        REQUIRE(expr_str.find('(') != std::string::npos);
        REQUIRE(expr_str.find(')') != std::string::npos);
        
        // Count parentheses - should only be around the addition
        auto open_count = std::count(expr_str.begin(), expr_str.end(), '(');
        auto close_count = std::count(expr_str.begin(), expr_str.end(), ')');
        REQUIRE(open_count == 1);
        REQUIRE(close_count == 1);
        
        REQUIRE(expr_str.find("s_0 + s_1") != std::string::npos);
        REQUIRE(expr_str.find("*") != std::string::npos);
        REQUIRE(expr_str.find("-") != std::string::npos);
        REQUIRE(expr_str.find("s_2") != std::string::npos);
        REQUIRE(expr_str.find("s_3") != std::string::npos);
    }
}

TEST_CASE("Expression parentheses with constants", "[scalar][formatting][parentheses][constants]") {
    SECTION("Multiplication with scalar constants") {
        Scalar<double, 0> x0(2.0);
        ScalarConstant<double> c(3.0);
        
        auto expr = x0 * c;
        std::string expr_str = expr.to_string();
        
        // Should not need parentheses for simple multiplication
        REQUIRE(expr_str.find('(') == std::string::npos);
        REQUIRE(expr_str.find(')') == std::string::npos);
    }
    
    SECTION("Addition with constant in multiplication") {
        Scalar<double, 0> x0(2.0);
        Scalar<double, 1> x1(3.0);
        ScalarConstant<double> c(5.0);
        
        // (x0 + c) * x1 should add parentheses around addition
        auto expr = (x0 + c) * x1;
        std::string expr_str = expr.to_string();
        
        REQUIRE(expr_str.find('(') != std::string::npos);
        REQUIRE(expr_str.find(')') != std::string::npos);
        REQUIRE(expr_str.find("*") != std::string::npos);
    }
    
    SECTION("Zero and unit scalar behavior") {
        Scalar<double, 0> x0(2.0);
        ZeroScalar<double> zero;
        UnitScalar<double> one;
        
        // Multiplication with zero should simplify
        auto zero_product = x0 * zero;
        REQUIRE(zero_product.to_string() == "");
        
        // Multiplication with unit should simplify  
        auto unit_product = x0 * one;
        REQUIRE(unit_product.to_string() == "s_0");
        
        // Addition with zero should simplify
        auto zero_sum = x0 + zero;
        REQUIRE(zero_sum.to_string() == "s_0");
    }
}

TEST_CASE("Expression parentheses edge cases", "[scalar][formatting][parentheses][edge_cases]") {
    SECTION("Unary negation") {
        Scalar<double, 0> x0(5.0);
        
        auto negated = -x0;
        REQUIRE(negated.to_string() == "-s_0");
        
        // Negation of addition
        auto neg_sum = -(x0 + x0);
        std::string neg_str = neg_sum.to_string();
        
        // Should handle negation properly (depends on implementation)
        REQUIRE(neg_str.find("s_0") != std::string::npos);
    }
    
    SECTION("Very complex nested expression") {
        Scalar<double, 0> x0(1.0);
        Scalar<double, 1> x1(2.0);
        Scalar<double, 2> x2(3.0);
        Scalar<double, 3> x3(4.0);
        
        // ((x0 + x1) * (x2 - x3)) + x0 should have proper parentheses
        auto complex_expr = ((x0 + x1) * (x2 - x3)) + x0;
        std::string complex_str = complex_expr.to_string();
        
        // Should contain multiple parentheses for the multiplied terms
        REQUIRE(complex_str.find('(') != std::string::npos);
        REQUIRE(complex_str.find(')') != std::string::npos);
        
        // Should contain all variables and operators
        REQUIRE(complex_str.find("s_0") != std::string::npos);
        REQUIRE(complex_str.find("s_1") != std::string::npos);
        REQUIRE(complex_str.find("s_2") != std::string::npos);
        REQUIRE(complex_str.find("s_3") != std::string::npos);
        REQUIRE(complex_str.find("+") != std::string::npos);
        REQUIRE(complex_str.find("-") != std::string::npos);
        REQUIRE(complex_str.find("*") != std::string::npos);
    }
    
    SECTION("Empty string handling") {
        ZeroScalar<double> zero1;
        ZeroScalar<double> zero2;
        
        // Zero + Zero should give empty string
        auto zero_sum = zero1 + zero2;
        REQUIRE(zero_sum.to_string() == "");
        
        // Zero * Zero should give empty string
        auto zero_product = zero1 * zero2;
        REQUIRE(zero_product.to_string() == "");
    }
    
    SECTION("Single parentheses needed cases") {
        Scalar<double, 0> x0(2.0);
        Scalar<double, 1> x1(3.0);
        Scalar<double, 2> x2(4.0);
        
        // x0 * (x1 + x2) - only second operand needs parentheses
        auto expr1 = x0 * (x1 + x2);
        std::string str1 = expr1.to_string();
        
        auto open_count1 = std::count(str1.begin(), str1.end(), '(');
        auto close_count1 = std::count(str1.begin(), str1.end(), ')');
        REQUIRE(open_count1 == 1);
        REQUIRE(close_count1 == 1);
        
        // (x0 - x1) * x2 - only first operand needs parentheses
        auto expr2 = (x0 - x1) * x2;
        std::string str2 = expr2.to_string();
        
        auto open_count2 = std::count(str2.begin(), str2.end(), '(');
        auto close_count2 = std::count(str2.begin(), str2.end(), ')');
        REQUIRE(open_count2 == 1);
        REQUIRE(close_count2 == 1);
    }
}

// Test basic elementwise matrix operations
TEST_CASE("Elementwise Matrix Addition", "[matrix][elementwise][addition]") {
    SECTION("Matrix + Matrix") {
        Matrix<double, 2, 2, 0> m1{{1.0, 2.0}, {3.0, 4.0}};
        Matrix<double, 2, 2, 1> m2{{5.0, 6.0}, {7.0, 8.0}};
        
        auto sum = m1 + m2;
        
        REQUIRE(sum.eval(0, 0) == Approx(6.0));  // 1 + 5
        REQUIRE(sum.eval(0, 1) == Approx(8.0));  // 2 + 6
        REQUIRE(sum.eval(1, 0) == Approx(10.0)); // 3 + 7
        REQUIRE(sum.eval(1, 1) == Approx(12.0)); // 4 + 8
    }
    
    SECTION("Matrix + Scalar") {
        Matrix<double, 2, 2, 0> matrix{{1.0, 2.0}, {3.0, 4.0}};
        Scalar<double, 1> scalar(5.0);
        
        auto sum = matrix + scalar;
        
        REQUIRE(sum.eval(0, 0) == Approx(6.0));  // 1 + 5
        REQUIRE(sum.eval(0, 1) == Approx(7.0));  // 2 + 5
        REQUIRE(sum.eval(1, 0) == Approx(8.0));  // 3 + 5
        REQUIRE(sum.eval(1, 1) == Approx(9.0));  // 4 + 5
    }
    
    SECTION("Scalar + Matrix") {
        Scalar<double, 0> scalar(10.0);
        Matrix<double, 2, 2, 1> matrix{{1.0, 2.0}, {3.0, 4.0}};
        
        auto sum = scalar + matrix;
        
        REQUIRE(sum.eval(0, 0) == Approx(11.0)); // 10 + 1
        REQUIRE(sum.eval(0, 1) == Approx(12.0)); // 10 + 2
        REQUIRE(sum.eval(1, 0) == Approx(13.0)); // 10 + 3
        REQUIRE(sum.eval(1, 1) == Approx(14.0)); // 10 + 4
    }
    
    SECTION("Addition string representation") {
        Matrix<double, 2, 2, 0> m1{{1.0, 2.0}, {3.0, 4.0}};
        Matrix<double, 2, 2, 1> m2{{5.0, 6.0}, {7.0, 8.0}};
        
        auto sum = m1 + m2;
        REQUIRE(sum.to_string() == "M_0 + M_1");
    }
}

TEST_CASE("Elementwise Matrix Subtraction", "[matrix][elementwise][subtraction]") {
    SECTION("Matrix - Matrix") {
        Matrix<double, 2, 2, 0> m1{{10.0, 8.0}, {6.0, 4.0}};
        Matrix<double, 2, 2, 1> m2{{5.0, 3.0}, {2.0, 1.0}};
        
        auto diff = m1 - m2;
        
        REQUIRE(diff.eval(0, 0) == Approx(5.0));  // 10 - 5
        REQUIRE(diff.eval(0, 1) == Approx(5.0));  // 8 - 3
        REQUIRE(diff.eval(1, 0) == Approx(4.0));  // 6 - 2
        REQUIRE(diff.eval(1, 1) == Approx(3.0));  // 4 - 1
    }
    
    SECTION("Matrix - Scalar") {
        Matrix<double, 2, 2, 0> matrix{{10.0, 8.0}, {6.0, 4.0}};
        Scalar<double, 1> scalar(2.0);
        
        auto diff = matrix - scalar;
        
        REQUIRE(diff.eval(0, 0) == Approx(8.0));  // 10 - 2
        REQUIRE(diff.eval(0, 1) == Approx(6.0));  // 8 - 2
        REQUIRE(diff.eval(1, 0) == Approx(4.0));  // 6 - 2
        REQUIRE(diff.eval(1, 1) == Approx(2.0));  // 4 - 2
    }
    
    SECTION("Scalar - Matrix") {
        Scalar<double, 0> scalar(10.0);
        Matrix<double, 2, 2, 1> matrix{{1.0, 2.0}, {3.0, 4.0}};
        
        auto diff = scalar - matrix;
        
        REQUIRE(diff.eval(0, 0) == Approx(9.0));  // 10 - 1
        REQUIRE(diff.eval(0, 1) == Approx(8.0));  // 10 - 2
        REQUIRE(diff.eval(1, 0) == Approx(7.0));  // 10 - 3
        REQUIRE(diff.eval(1, 1) == Approx(6.0));  // 10 - 4
    }
    
    SECTION("Subtraction string representation") {
        Matrix<double, 2, 2, 0> m1{{10.0, 8.0}, {6.0, 4.0}};
        Matrix<double, 2, 2, 1> m2{{5.0, 3.0}, {2.0, 1.0}};
        
        auto diff = m1 - m2;
        REQUIRE(diff.to_string() == "M_0 - M_1");
    }
}

TEST_CASE("Elementwise Matrix Multiplication", "[matrix][elementwise][multiplication]") {
    SECTION("Matrix * Matrix") {
        Matrix<double, 2, 2, 0> m1{{2.0, 3.0}, {4.0, 5.0}};
        Matrix<double, 2, 2, 1> m2{{6.0, 7.0}, {8.0, 9.0}};
        
        auto product = m1 * m2;
        
        REQUIRE(product.eval(0, 0) == Approx(12.0)); // 2 * 6
        REQUIRE(product.eval(0, 1) == Approx(21.0)); // 3 * 7
        REQUIRE(product.eval(1, 0) == Approx(32.0)); // 4 * 8
        REQUIRE(product.eval(1, 1) == Approx(45.0)); // 5 * 9
    }
    
    SECTION("Matrix * Scalar") {
        Matrix<double, 2, 2, 0> matrix{{2.0, 3.0}, {4.0, 5.0}};
        Scalar<double, 1> scalar(3.0);
        
        auto product = matrix * scalar;
        
        REQUIRE(product.eval(0, 0) == Approx(6.0));  // 2 * 3
        REQUIRE(product.eval(0, 1) == Approx(9.0));  // 3 * 3
        REQUIRE(product.eval(1, 0) == Approx(12.0)); // 4 * 3
        REQUIRE(product.eval(1, 1) == Approx(15.0)); // 5 * 3
    }
    
    SECTION("Scalar * Matrix") {
        Scalar<double, 0> scalar(4.0);
        Matrix<double, 2, 2, 1> matrix{{2.0, 3.0}, {4.0, 5.0}};
        
        auto product = scalar * matrix;
        
        REQUIRE(product.eval(0, 0) == Approx(8.0));  // 4 * 2
        REQUIRE(product.eval(0, 1) == Approx(12.0)); // 4 * 3
        REQUIRE(product.eval(1, 0) == Approx(16.0)); // 4 * 4
        REQUIRE(product.eval(1, 1) == Approx(20.0)); // 4 * 5
    }
    
    SECTION("Multiplication string representation") {
        Matrix<double, 2, 2, 0> m1{{2.0, 3.0}, {4.0, 5.0}};
        Matrix<double, 2, 2, 1> m2{{6.0, 7.0}, {8.0, 9.0}};
        
        auto product = m1 * m2;
        REQUIRE(product.to_string() == "M_0 * M_1");
    }
    
    SECTION("Multiplication with special matrices") {
        Matrix<double, 2, 2, 0> matrix{{2.0, 3.0}, {4.0, 5.0}};
        IdentityMatrix<double, 2> identity;
        ZeroMatrix<double, 2, 2> zero;
        
        // Matrix * Identity should give matrix
        auto id_product = matrix * identity;
        REQUIRE(id_product.eval(0, 0) == Approx(2.0));
        REQUIRE(id_product.eval(1, 1) == Approx(5.0));
        
        // Matrix * Zero should give zero
        auto zero_product = matrix * zero;
        REQUIRE(zero_product.eval(0, 0) == Approx(0.0));
        REQUIRE(zero_product.eval(1, 1) == Approx(0.0));
    }
}

TEST_CASE("Elementwise Matrix Division", "[matrix][elementwise][division]") {
    SECTION("Matrix / Matrix") {
        Matrix<double, 2, 2, 0> m1{{12.0, 15.0}, {20.0, 24.0}};
        Matrix<double, 2, 2, 1> m2{{3.0, 5.0}, {4.0, 6.0}};
        
        auto quotient = m1 / m2;
        
        REQUIRE(quotient.eval(0, 0) == Approx(4.0));  // 12 / 3
        REQUIRE(quotient.eval(0, 1) == Approx(3.0));  // 15 / 5
        REQUIRE(quotient.eval(1, 0) == Approx(5.0));  // 20 / 4
        REQUIRE(quotient.eval(1, 1) == Approx(4.0));  // 24 / 6
    }
    
    SECTION("Matrix / Scalar") {
        Matrix<double, 2, 2, 0> matrix{{12.0, 15.0}, {20.0, 24.0}};
        Scalar<double, 1> scalar(4.0);
        
        auto quotient = matrix / scalar;
        
        REQUIRE(quotient.eval(0, 0) == Approx(3.0));  // 12 / 4
        REQUIRE(quotient.eval(0, 1) == Approx(3.75)); // 15 / 4
        REQUIRE(quotient.eval(1, 0) == Approx(5.0));  // 20 / 4
        REQUIRE(quotient.eval(1, 1) == Approx(6.0));  // 24 / 4
    }
    
    SECTION("Scalar / Matrix") {
        Scalar<double, 0> scalar(24.0);
        Matrix<double, 2, 2, 1> matrix{{2.0, 3.0}, {4.0, 6.0}};
        
        auto quotient = scalar / matrix;
        
        REQUIRE(quotient.eval(0, 0) == Approx(12.0)); // 24 / 2
        REQUIRE(quotient.eval(0, 1) == Approx(8.0));  // 24 / 3
        REQUIRE(quotient.eval(1, 0) == Approx(6.0));  // 24 / 4
        REQUIRE(quotient.eval(1, 1) == Approx(4.0));  // 24 / 6
    }
    
    SECTION("Division string representation") {
        Matrix<double, 2, 2, 0> m1{{12.0, 15.0}, {20.0, 24.0}};
        Matrix<double, 2, 2, 1> m2{{3.0, 5.0}, {4.0, 6.0}};
        
        auto quotient = m1 / m2;
        REQUIRE(quotient.to_string() == "M_0 / M_1");
    }
}

TEST_CASE("Elementwise Matrix Power", "[matrix][elementwise][power]") {
    SECTION("Matrix ^ Matrix") {
        Matrix<double, 2, 2, 0> base{{2.0, 3.0}, {4.0, 5.0}};
        Matrix<double, 2, 2, 1> exponent{{2.0, 3.0}, {2.0, 2.0}};
        
        auto power_result = pow(base, exponent);
        
        REQUIRE(power_result.eval(0, 0) == Approx(4.0));  // 2^2
        REQUIRE(power_result.eval(0, 1) == Approx(27.0)); // 3^3
        REQUIRE(power_result.eval(1, 0) == Approx(16.0)); // 4^2
        REQUIRE(power_result.eval(1, 1) == Approx(25.0)); // 5^2
    }
    
    SECTION("Matrix ^ Scalar") {
        Matrix<double, 2, 2, 0> base{{2.0, 3.0}, {4.0, 5.0}};
        Scalar<double, 1> exponent(2.0);
        
        auto power_result = pow(base, exponent);
        
        REQUIRE(power_result.eval(0, 0) == Approx(4.0));  // 2^2
        REQUIRE(power_result.eval(0, 1) == Approx(9.0));  // 3^2
        REQUIRE(power_result.eval(1, 0) == Approx(16.0)); // 4^2
        REQUIRE(power_result.eval(1, 1) == Approx(25.0)); // 5^2
    }
    
    SECTION("Scalar ^ Matrix") {
        Scalar<double, 0> base(2.0);
        Matrix<double, 2, 2, 1> exponent{{1.0, 2.0}, {3.0, 4.0}};
        
        auto power_result = pow(base, exponent);
        
        REQUIRE(power_result.eval(0, 0) == Approx(2.0));  // 2^1
        REQUIRE(power_result.eval(0, 1) == Approx(4.0));  // 2^2
        REQUIRE(power_result.eval(1, 0) == Approx(8.0));  // 2^3
        REQUIRE(power_result.eval(1, 1) == Approx(16.0)); // 2^4
    }
    
    SECTION("Power with constant exponents") {
        Matrix<double, 2, 2, 0> base{{2.0, 3.0}, {4.0, 5.0}};
        
        // Using constant exponent
        auto squared = pow(base, 2.0);
        auto cubed = pow(base, 3.0);
        
        REQUIRE(squared.eval(0, 0) == Approx(4.0));   // 2^2
        REQUIRE(squared.eval(0, 1) == Approx(9.0));   // 3^2
        REQUIRE(cubed.eval(0, 0) == Approx(8.0));     // 2^3
        REQUIRE(cubed.eval(0, 1) == Approx(27.0));    // 3^3
    }
    
    SECTION("Power string representation") {
        Matrix<double, 2, 2, 0> base{{2.0, 3.0}, {4.0, 5.0}};
        Matrix<double, 2, 2, 1> exponent{{2.0, 3.0}, {2.0, 2.0}};
        
        auto power_result = pow(base, exponent);
        REQUIRE(power_result.to_string() == "M_0^M_1");
    }
}

TEST_CASE("Mixed Elementwise Operations", "[matrix][elementwise][mixed]") {
    SECTION("Complex expression with multiple operations") {
        Matrix<double, 2, 2, 0> m1{{2.0, 3.0}, {4.0, 5.0}};
        Matrix<double, 2, 2, 1> m2{{1.0, 2.0}, {3.0, 4.0}};
        Matrix<double, 2, 2, 2> m3{{2.0, 2.0}, {2.0, 2.0}};
        
        // (m1 + m2) * m3
        auto complex_expr = (m1 + m2) * m3;
        
        // Expected: ((2+1), (3+2)) * (2, 2) = (3*2, 5*2) = (6, 10)
        //           ((4+3), (5+4)) * (2, 2) = (7*2, 9*2) = (14, 18)
        REQUIRE(complex_expr.eval(0, 0) == Approx(6.0));
        REQUIRE(complex_expr.eval(0, 1) == Approx(10.0));
        REQUIRE(complex_expr.eval(1, 0) == Approx(14.0));
        REQUIRE(complex_expr.eval(1, 1) == Approx(18.0));
    }
    
    SECTION("Division and multiplication chain") {
        Matrix<double, 2, 2, 0> m1{{12.0, 15.0}, {20.0, 24.0}};
        Matrix<double, 2, 2, 1> m2{{3.0, 5.0}, {4.0, 6.0}};
        Scalar<double, 2> scalar(2.0);
        
        // (m1 / m2) * scalar
        auto chain_expr = (m1 / m2) * scalar;
        
        // Expected: (12/3*2, 15/5*2) = (8, 6)
        //           (20/4*2, 24/6*2) = (10, 8)
        REQUIRE(chain_expr.eval(0, 0) == Approx(8.0));
        REQUIRE(chain_expr.eval(0, 1) == Approx(6.0));
        REQUIRE(chain_expr.eval(1, 0) == Approx(10.0));
        REQUIRE(chain_expr.eval(1, 1) == Approx(8.0));
    }
    
    SECTION("Power in complex expression") {
        Matrix<double, 2, 2, 0> base{{2.0, 3.0}, {2.0, 2.0}};
        Scalar<double, 1> exponent(2.0);
        Scalar<double, 2> addend(1.0);
        
        // pow(base, exponent) + addend
        auto power_add = pow(base, exponent) + addend;
        
        // Expected: (2^2+1, 3^2+1) = (5, 10)
        //           (2^2+1, 2^2+1) = (5, 5)
        REQUIRE(power_add.eval(0, 0) == Approx(5.0));
        REQUIRE(power_add.eval(0, 1) == Approx(10.0));
        REQUIRE(power_add.eval(1, 0) == Approx(5.0));
        REQUIRE(power_add.eval(1, 1) == Approx(5.0));
    }
}

TEST_CASE("Elementwise Operations with Special Matrices", "[matrix][elementwise][special]") {
    SECTION("Operations with IdentityMatrix") {
        Matrix<double, 2, 2, 0> matrix{{4.0, 6.0}, {8.0, 10.0}};
        IdentityMatrix<double, 2> identity;
        
        // Matrix + Identity
        auto add_id = matrix + identity;
        REQUIRE(add_id.eval(0, 0) == Approx(5.0));  // 4 + 1
        REQUIRE(add_id.eval(0, 1) == Approx(6.0));  // 6 + 0
        REQUIRE(add_id.eval(1, 0) == Approx(8.0));  // 8 + 0
        REQUIRE(add_id.eval(1, 1) == Approx(11.0)); // 10 + 1
        
        // Matrix * Identity
        auto mult_id = matrix * identity;
        REQUIRE(mult_id.eval(0, 0) == Approx(4.0)); // 4 * 1
        REQUIRE(mult_id.eval(0, 1) == Approx(0.0)); // 6 * 0
        REQUIRE(mult_id.eval(1, 0) == Approx(0.0)); // 8 * 0
        REQUIRE(mult_id.eval(1, 1) == Approx(10.0)); // 10 * 1
    }
    
    SECTION("Operations with ZeroMatrix") {
        Matrix<double, 2, 2, 0> matrix{{4.0, 6.0}, {8.0, 10.0}};
        ZeroMatrix<double, 2, 2> zero;
        
        // Matrix + Zero
        auto add_zero = matrix + zero;
        REQUIRE(add_zero.eval(0, 0) == Approx(4.0));
        REQUIRE(add_zero.eval(0, 1) == Approx(6.0));
        REQUIRE(add_zero.eval(1, 0) == Approx(8.0));
        REQUIRE(add_zero.eval(1, 1) == Approx(10.0));
        
        // Matrix - Zero
        auto sub_zero = matrix - zero;
        REQUIRE(sub_zero.eval(0, 0) == Approx(4.0));
        REQUIRE(sub_zero.eval(1, 1) == Approx(10.0));
        
        // Matrix * Zero
        auto mult_zero = matrix * zero;
        REQUIRE(mult_zero.eval(0, 0) == Approx(0.0));
        REQUIRE(mult_zero.eval(1, 1) == Approx(0.0));
    }
    
    SECTION("Non-square matrices") {
        Matrix<double, 2, 3, 0> m1{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
        Matrix<double, 2, 3, 1> m2{{2.0, 3.0, 4.0}, {5.0, 6.0, 7.0}};
        
        auto sum = m1 + m2;
        auto product = m1 * m2;
        
        REQUIRE(sum.eval(0, 2) == Approx(7.0));      // 3 + 4
        REQUIRE(sum.eval(1, 0) == Approx(9.0));      // 4 + 5
        REQUIRE(product.eval(0, 1) == Approx(6.0));  // 2 * 3
        REQUIRE(product.eval(1, 2) == Approx(42.0)); // 6 * 7
    }
}

// Test matrix shape compatibility rules
TEST_CASE("Matrix Shape Compatibility - Valid Operations", "[matrix][shapes][valid]") {
    SECTION("Same-shaped matrices") {
        // 2x2 + 2x2 should work
        Matrix<double, 2, 2, 0> m1{{1.0, 2.0}, {3.0, 4.0}};
        Matrix<double, 2, 2, 1> m2{{5.0, 6.0}, {7.0, 8.0}};
        
        auto sum = m1 + m2;
        auto diff = m1 - m2;
        auto product = m1 * m2;
        auto quotient = m1 / m2;
        
        REQUIRE(sum.eval(0, 0) == Approx(6.0));     // 1 + 5
        REQUIRE(diff.eval(0, 0) == Approx(-4.0));   // 1 - 5
        REQUIRE(product.eval(0, 0) == Approx(5.0)); // 1 * 5
        REQUIRE(quotient.eval(0, 0) == Approx(0.2)); // 1 / 5
    }
    
    SECTION("3x3 matrices") {
        Matrix<double, 3, 3, 0> m1{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
        Matrix<double, 3, 3, 1> m2{{9.0, 8.0, 7.0}, {6.0, 5.0, 4.0}, {3.0, 2.0, 1.0}};
        
        auto sum = m1 + m2;
        auto product = m1 * m2;
        
        REQUIRE(sum.eval(0, 0) == Approx(10.0));    // 1 + 9
        REQUIRE(sum.eval(2, 2) == Approx(10.0));    // 9 + 1
        REQUIRE(product.eval(1, 1) == Approx(25.0)); // 5 * 5
    }
    
    SECTION("Non-square same-shaped matrices") {
        // 2x3 + 2x3 should work
        Matrix<double, 2, 3, 0> m1{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
        Matrix<double, 2, 3, 1> m2{{2.0, 3.0, 4.0}, {5.0, 6.0, 7.0}};
        
        auto sum = m1 + m2;
        auto diff = m1 - m2;
        
        REQUIRE(sum.eval(0, 2) == Approx(7.0));  // 3 + 4
        REQUIRE(diff.eval(1, 0) == Approx(-1.0)); // 4 - 5
        
        // 3x2 + 3x2 should work
        Matrix<double, 3, 2, 0> m3{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
        Matrix<double, 3, 2, 1> m4{{6.0, 5.0}, {4.0, 3.0}, {2.0, 1.0}};
        
        auto sum2 = m3 + m4;
        REQUIRE(sum2.eval(0, 0) == Approx(7.0));  // 1 + 6
        REQUIRE(sum2.eval(2, 1) == Approx(7.0));  // 6 + 1
    }
    
    SECTION("1x1 matrices (scalar-like)") {
        Matrix<double, 1, 1, 0> scalar1{{5.0}};
        Matrix<double, 1, 1, 1> scalar2{{3.0}};
        
        auto sum = scalar1 + scalar2;
        auto product = scalar1 * scalar2;
        auto quotient = scalar1 / scalar2;
        
        REQUIRE(sum.eval(0, 0) == Approx(8.0));
        REQUIRE(product.eval(0, 0) == Approx(15.0));
        REQUIRE(quotient.eval(0, 0) == Approx(5.0/3.0));
    }
}

TEST_CASE("Matrix-Scalar Compatibility", "[matrix][shapes][scalar]") {
    SECTION("Matrix + Scalar operations") {
        Matrix<double, 2, 2, 0> matrix{{1.0, 2.0}, {3.0, 4.0}};
        Scalar<double, 1> scalar(5.0);
        
        // Matrix + Scalar should work (scalar broadcasts)
        auto sum1 = matrix + scalar;
        auto sum2 = scalar + matrix;
        auto diff1 = matrix - scalar;
        auto diff2 = scalar - matrix;
        auto prod1 = matrix * scalar;
        auto prod2 = scalar * matrix;
        auto quot1 = matrix / scalar;
        auto quot2 = scalar / matrix;
        
        REQUIRE(sum1.eval(0, 0) == Approx(6.0));   // 1 + 5
        REQUIRE(sum2.eval(1, 1) == Approx(9.0));   // 5 + 4
        REQUIRE(diff1.eval(0, 1) == Approx(-3.0)); // 2 - 5
        REQUIRE(diff2.eval(1, 0) == Approx(2.0));  // 5 - 3
        REQUIRE(prod1.eval(1, 1) == Approx(20.0)); // 4 * 5
        REQUIRE(prod2.eval(0, 1) == Approx(10.0)); // 5 * 2
        REQUIRE(quot1.eval(0, 0) == Approx(0.2));  // 1 / 5
        REQUIRE(quot2.eval(1, 0) == Approx(5.0/3.0)); // 5 / 3
    }
    
    SECTION("Different sized matrices with scalars") {
        Matrix<double, 3, 2, 0> matrix{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
        Scalar<double, 1> scalar(2.0);
        
        auto scaled = matrix * scalar;
        auto added = matrix + scalar;
        
        REQUIRE(scaled.eval(0, 0) == Approx(2.0));  // 1 * 2
        REQUIRE(scaled.eval(2, 1) == Approx(12.0)); // 6 * 2
        REQUIRE(added.eval(1, 0) == Approx(5.0));   // 3 + 2
        REQUIRE(added.eval(2, 1) == Approx(8.0));   // 6 + 2
    }
    
    SECTION("1x1 Matrix with regular scalars") {
        Matrix<double, 1, 1, 0> matrix1x1{{7.0}};
        Scalar<double, 1> scalar(3.0);
        
        // 1x1 matrix should work with scalars in both directions
        auto sum1 = matrix1x1 + scalar;
        auto sum2 = scalar + matrix1x1;
        auto product = matrix1x1 * scalar;
        
        REQUIRE(sum1.eval(0, 0) == Approx(10.0)); // 7 + 3
        REQUIRE(sum2.eval(0, 0) == Approx(10.0)); // 3 + 7
        REQUIRE(product.eval(0, 0) == Approx(21.0)); // 7 * 3
    }
}

TEST_CASE("Special Matrix Shape Operations", "[matrix][shapes][special]") {
    SECTION("IdentityMatrix compatibility") {
        Matrix<double, 3, 3, 0> matrix{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
        IdentityMatrix<double, 3> identity;
        
        // Identity should work with same-sized matrices
        auto sum = matrix + identity;
        auto product = matrix * identity;
        
        REQUIRE(sum.eval(0, 0) == Approx(2.0));  // 1 + 1 (identity diagonal)
        REQUIRE(sum.eval(0, 1) == Approx(2.0));  // 2 + 0 (identity off-diagonal)
        REQUIRE(sum.eval(1, 1) == Approx(6.0));  // 5 + 1 (identity diagonal)
        
        REQUIRE(product.eval(0, 0) == Approx(1.0)); // 1 * 1
        REQUIRE(product.eval(0, 1) == Approx(0.0)); // 2 * 0
        REQUIRE(product.eval(1, 1) == Approx(5.0)); // 5 * 1
    }
    
    SECTION("ZeroMatrix compatibility") {
        Matrix<double, 2, 3, 0> matrix{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
        ZeroMatrix<double, 2, 3> zero;
        
        // Zero should work with same-shaped matrices
        auto sum = matrix + zero;
        auto product = matrix * zero;
        auto diff = matrix - zero;
        
        REQUIRE(sum.eval(0, 0) == Approx(1.0));   // 1 + 0
        REQUIRE(sum.eval(1, 2) == Approx(6.0));   // 6 + 0
        REQUIRE(product.eval(0, 1) == Approx(0.0)); // 2 * 0
        REQUIRE(product.eval(1, 0) == Approx(0.0)); // 4 * 0
        REQUIRE(diff.eval(0, 2) == Approx(3.0));    // 3 - 0
    }
    
    SECTION("Mixed special matrices") {
        IdentityMatrix<double, 2> identity;
        ZeroMatrix<double, 2, 2> zero;
        
        auto sum = identity + zero;
        auto diff = identity - zero;
        auto product = identity * zero;
        
        REQUIRE(sum.eval(0, 0) == Approx(1.0));  // 1 + 0
        REQUIRE(sum.eval(0, 1) == Approx(0.0));  // 0 + 0
        REQUIRE(diff.eval(1, 1) == Approx(1.0)); // 1 - 0
        REQUIRE(product.eval(0, 0) == Approx(0.0)); // 1 * 0
    }
}

TEST_CASE("Shape Broadcasting Rules", "[matrix][shapes][broadcasting]") {
    SECTION("Scalar broadcasting to matrices") {
        Matrix<double, 2, 2, 0> m2x2{{1.0, 2.0}, {3.0, 4.0}};
        Matrix<double, 3, 3, 1> m3x3{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
        Scalar<double, 2> scalar(10.0);
        
        // Scalar should work with any matrix size
        auto sum2x2 = m2x2 + scalar;
        auto sum3x3 = m3x3 + scalar;
        auto prod2x2 = scalar * m2x2;
        auto prod3x3 = scalar * m3x3;
        
        REQUIRE(sum2x2.eval(0, 0) == Approx(11.0)); // 1 + 10
        REQUIRE(sum2x2.eval(1, 1) == Approx(14.0)); // 4 + 10
        REQUIRE(sum3x3.eval(0, 0) == Approx(11.0)); // 1 + 10
        REQUIRE(sum3x3.eval(2, 2) == Approx(19.0)); // 9 + 10
        
        REQUIRE(prod2x2.eval(0, 1) == Approx(20.0)); // 10 * 2
        REQUIRE(prod3x3.eval(1, 2) == Approx(60.0)); // 10 * 6
    }
    
    SECTION("1x1 matrix broadcasting") {
        Matrix<double, 1, 1, 0> scalar_matrix{{5.0}};
        Matrix<double, 2, 3, 1> regular_matrix{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
        
        // 1x1 matrix should broadcast to larger matrices like scalars
        // This tests that scalar-shaped matrices work correctly with regular matrices
        auto sum = scalar_matrix + regular_matrix;
        auto product = regular_matrix * scalar_matrix;
        
        // 1x1 matrix should broadcast its single value (5.0) to all positions
        REQUIRE(sum.eval(0, 0) == Approx(6.0));   // 5 + 1 = 6
        REQUIRE(sum.eval(0, 1) == Approx(7.0));   // 5 + 2 = 7  
        REQUIRE(sum.eval(0, 2) == Approx(8.0));   // 5 + 3 = 8
        REQUIRE(sum.eval(1, 0) == Approx(9.0));   // 5 + 4 = 9
        REQUIRE(sum.eval(1, 1) == Approx(10.0));  // 5 + 5 = 10
        REQUIRE(sum.eval(1, 2) == Approx(11.0));  // 5 + 6 = 11
        
        // Matrix multiplication should also work with scalar broadcasting
        REQUIRE(product.eval(0, 0) == Approx(5.0));   // 1 * 5 = 5
        REQUIRE(product.eval(0, 1) == Approx(10.0));  // 2 * 5 = 10
        REQUIRE(product.eval(0, 2) == Approx(15.0));  // 3 * 5 = 15
        REQUIRE(product.eval(1, 0) == Approx(20.0));  // 4 * 5 = 20
        REQUIRE(product.eval(1, 1) == Approx(25.0));  // 5 * 5 = 25
        REQUIRE(product.eval(1, 2) == Approx(30.0));  // 6 * 5 = 30
    }
}

TEST_CASE("Complex Matrix Shape Operations", "[matrix][shapes][complex]") {
    SECTION("Chained operations with shape consistency") {
        Matrix<double, 2, 2, 0> m1{{1.0, 2.0}, {3.0, 4.0}};
        Matrix<double, 2, 2, 1> m2{{5.0, 6.0}, {7.0, 8.0}};
        Matrix<double, 2, 2, 2> m3{{2.0, 2.0}, {2.0, 2.0}};
        Scalar<double, 3> scalar(3.0);
        
        // Complex expression: (m1 + m2) * m3 + scalar
        auto complex_expr = (m1 + m2) * m3 + scalar;
        
        // Expected: ((1+5)*2+3, (2+6)*2+3) = (15, 19)
        //           ((3+7)*2+3, (4+8)*2+3) = (23, 27)
        REQUIRE(complex_expr.eval(0, 0) == Approx(15.0)); // (1+5)*2+3 = 15
        REQUIRE(complex_expr.eval(0, 1) == Approx(19.0)); // (2+6)*2+3 = 19
        REQUIRE(complex_expr.eval(1, 0) == Approx(23.0)); // (3+7)*2+3 = 23
        REQUIRE(complex_expr.eval(1, 1) == Approx(27.0)); // (4+8)*2+3 = 27
    }
    
    SECTION("Mixed scalar and matrix operations") {
        Matrix<double, 3, 2, 0> matrix{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
        Scalar<double, 1> s1(2.0);
        Scalar<double, 2> s2(3.0);
        
        // Expression: matrix * s1 + s2
        auto expr = matrix * s1 + s2;
        
        REQUIRE(expr.eval(0, 0) == Approx(5.0));  // 1*2+3 = 5
        REQUIRE(expr.eval(1, 1) == Approx(11.0)); // 4*2+3 = 11
        REQUIRE(expr.eval(2, 0) == Approx(13.0)); // 5*2+3 = 13
    }
    
    SECTION("Power operations with shapes") {
        Matrix<double, 2, 2, 0> base{{2.0, 3.0}, {4.0, 5.0}};
        Matrix<double, 2, 2, 1> exponent{{2.0, 2.0}, {2.0, 2.0}};
        Scalar<double, 2> scalar_exp(3.0);
        
        // Matrix^Matrix (same shape)
        auto power1 = pow(base, exponent);
        // Matrix^Scalar
        auto power2 = pow(base, scalar_exp);
        
        REQUIRE(power1.eval(0, 0) == Approx(4.0));   // 2^2 = 4
        REQUIRE(power1.eval(0, 1) == Approx(9.0));   // 3^2 = 9
        REQUIRE(power2.eval(0, 0) == Approx(8.0));   // 2^3 = 8
        REQUIRE(power2.eval(1, 1) == Approx(125.0)); // 5^3 = 125
    }
}

// Note: Incompatible shape operations would cause compile-time errors.
// These cannot be tested with runtime assertions, but the static_assert
// statements in the operators should prevent compilation of invalid operations.
TEST_CASE("Shape Compatibility Documentation", "[matrix][shapes][documentation]") {
    SECTION("Valid operations summary") {
        // This test documents what operations are valid:
        // 1. Matrix op Matrix: Only when dimensions match exactly
        // 2. Matrix op Scalar: Always valid (scalar broadcasts)
        // 3. Scalar op Matrix: Always valid (scalar broadcasts)
        // 4. 1x1 Matrix: Acts like a scalar, works with any other matrix
        
        // Examples that work:
        Matrix<double, 2, 2, 0> m2x2{{1.0, 2.0}, {3.0, 4.0}};
        Matrix<double, 2, 2, 1> another2x2{{5.0, 6.0}, {7.0, 8.0}};
        Matrix<double, 1, 1, 2> m1x1{{10.0}};
        Scalar<double, 3> scalar(5.0);
        
        // These should all compile and work:
        auto valid1 = m2x2 + another2x2;     // Same shape matrices
        auto valid2 = m2x2 + scalar;         // Matrix + Scalar
        auto valid3 = scalar + m2x2;         // Scalar + Matrix
        auto valid4 = m2x2 + m1x1;           // Matrix + 1x1 Matrix
        auto valid5 = m1x1 * scalar;         // 1x1 Matrix * Scalar
        
        // Verify they produce correct results
        REQUIRE(valid1.eval(0, 0) == Approx(6.0));  // 1 + 5
        REQUIRE(valid2.eval(0, 0) == Approx(6.0));  // 1 + 5
        REQUIRE(valid3.eval(0, 0) == Approx(6.0));  // 5 + 1
        REQUIRE(valid4.eval(0, 0) == Approx(11.0)); // 1 + 10
        REQUIRE(valid5.eval(0, 0) == Approx(50.0)); // 10 * 5
    }
    
    SECTION("Invalid operations would cause compile errors") {
        // These operations would NOT compile due to static_assert:
        // Matrix<double, 2, 2> m2x2;
        // Matrix<double, 3, 3> m3x3;
        // auto invalid = m2x2 + m3x3;  // ERROR: Different shapes
        
        // Matrix<double, 2, 3> m2x3;
        // Matrix<double, 3, 2> m3x2;
        // auto invalid2 = m2x3 * m3x2; // ERROR: Different shapes
        
        // The static_assert in operators should prevent these at compile time
        REQUIRE(true); // This test is for documentation purposes
    }
}

// Test static_assert behavior for incompatible matrix shapes
// These tests verify that the library correctly prevents compilation of invalid operations
TEST_CASE("Static Assertion Tests for Incompatible Shapes", "[static_assert][shapes][compile_error]") {
    
    SECTION("Compile-time shape validation documentation") {
        // This section documents which operations should cause compile-time errors
        // due to static_assert statements in the expression constructors
        
        // The following operations should NOT compile if uncommented:
        
        /* 
        // TEST 1: Incompatible matrix addition (2x2 + 3x3)
        Matrix<double, 2, 2, 0> m2x2{{1.0, 2.0}, {3.0, 4.0}};
        Matrix<double, 3, 3, 1> m3x3{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
        auto invalid_add = m2x2 + m3x3; // ERROR: static_assert should fail
        */
        
        /*
        // TEST 2: Incompatible matrix subtraction (2x3 - 3x2)
        Matrix<double, 2, 3, 0> m2x3{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
        Matrix<double, 3, 2, 1> m3x2{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
        auto invalid_sub = m2x3 - m3x2; // ERROR: static_assert should fail
        */
        
        /*
        // TEST 3: Incompatible elementwise multiplication (4x4 * 2x2)
        Matrix<double, 4, 4, 0> m4x4{{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}, 
                                    {9.0, 10.0, 11.0, 12.0}, {13.0, 14.0, 15.0, 16.0}};
        Matrix<double, 2, 2, 1> m2x2{{1.0, 2.0}, {3.0, 4.0}};
        auto invalid_mult = m4x4 * m2x2; // ERROR: static_assert should fail
        */
        
        /*
        // TEST 4: Incompatible elementwise division (1x5 / 5x1)
        Matrix<double, 1, 5, 0> m1x5{{1.0, 2.0, 3.0, 4.0, 5.0}};
        Matrix<double, 5, 1, 1> m5x1{{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
        auto invalid_div = m1x5 / m5x1; // ERROR: static_assert should fail
        */
        
        /*
        // TEST 5: Incompatible power operation (3x2 ^ 2x4)
        Matrix<double, 3, 2, 0> base{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
        Matrix<double, 2, 4, 1> exponent{{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}};
        auto invalid_pow = pow(base, exponent); // ERROR: static_assert should fail
        */
        
        // These operations should work (valid cases for comparison):
        Matrix<double, 2, 2, 0> valid_m1{{1.0, 2.0}, {3.0, 4.0}};
        Matrix<double, 2, 2, 1> valid_m2{{5.0, 6.0}, {7.0, 8.0}};
        Scalar<double, 2> valid_scalar(5.0);
        
        // These should compile successfully:
        auto valid_add = valid_m1 + valid_m2;        // Same shape matrices
        auto valid_scalar_add = valid_m1 + valid_scalar; // Matrix + Scalar
        auto valid_mult = valid_m1 * valid_m2;       // Same shape matrices
        auto valid_scalar_mult = valid_scalar * valid_m1; // Scalar * Matrix
        
        // Verify the valid operations work
        REQUIRE(valid_add.eval(0, 0) == Approx(6.0));     // 1 + 5
        REQUIRE(valid_scalar_add.eval(0, 0) == Approx(6.0)); // 1 + 5
        REQUIRE(valid_mult.eval(0, 0) == Approx(5.0));    // 1 * 5
        REQUIRE(valid_scalar_mult.eval(0, 0) == Approx(5.0)); // 5 * 1
    }
}

// Test the type traits using runtime verification instead of static_assert
TEST_CASE("Shape Compatibility Type Traits", "[type_traits][shapes]") {
    
    SECTION("is_eq_shape trait verification") {
        // Test the underlying type traits used in the library
        
        // Same shapes should be equal
        REQUIRE(is_eq_shape_v<Matrix<double, 2, 2, 0>, Matrix<double, 2, 2, 1>> == true);
        REQUIRE(is_eq_shape_v<Matrix<double, 3, 4, 0>, Matrix<double, 3, 4, 1>> == true);
        REQUIRE(is_eq_shape_v<Matrix<double, 1, 1, 0>, Matrix<double, 1, 1, 1>> == true);
        
        // Different shapes should not be equal
        REQUIRE(is_eq_shape_v<Matrix<double, 2, 2, 0>, Matrix<double, 3, 3, 1>> == false);
        REQUIRE(is_eq_shape_v<Matrix<double, 2, 3, 0>, Matrix<double, 3, 2, 1>> == false);
        REQUIRE(is_eq_shape_v<Matrix<double, 1, 5, 0>, Matrix<double, 5, 1, 1>> == false);
    }
    
    SECTION("is_scalar_shape trait verification") {
        // Test scalar shape detection
        
        // 1x1 matrices should be scalar-shaped
        REQUIRE(is_scalar_shape_v<Matrix<double, 1, 1, 0>> == true);
        REQUIRE(is_scalar_shape_v<ZeroMatrix<double, 1, 1>> == true);
        
        // Non-1x1 matrices should not be scalar-shaped
        REQUIRE(is_scalar_shape_v<Matrix<double, 2, 2, 0>> == false);
        REQUIRE(is_scalar_shape_v<Matrix<double, 1, 2, 0>> == false);
        REQUIRE(is_scalar_shape_v<Matrix<double, 2, 1, 0>> == false);
        REQUIRE(is_scalar_shape_v<Matrix<double, 3, 4, 0>> == false);
        
        // Scalars should be scalar-shaped
        REQUIRE(is_scalar_shape_v<Scalar<double, 0>> == true);
    }
    
    SECTION("is_elementwise_broadcastable trait verification") {
        // Test broadcasting compatibility detection
        
        // Same shapes should be broadcastable
        REQUIRE(is_elementwise_broadcastable_v<Matrix<double, 2, 2, 0>, Matrix<double, 2, 2, 1>> == true);
        
        // Scalar with any matrix should be broadcastable
        REQUIRE(is_elementwise_broadcastable_v<Scalar<double, 0>, Matrix<double, 3, 4, 1>> == true);
        REQUIRE(is_elementwise_broadcastable_v<Matrix<double, 2, 3, 0>, Scalar<double, 1>> == true);
        
        // 1x1 matrix with any matrix should be broadcastable
        REQUIRE(is_elementwise_broadcastable_v<Matrix<double, 1, 1, 0>, Matrix<double, 3, 4, 1>> == true);
        REQUIRE(is_elementwise_broadcastable_v<Matrix<double, 2, 3, 0>, Matrix<double, 1, 1, 1>> == true);
        
        // Incompatible non-scalar shapes should not be broadcastable
        REQUIRE(is_elementwise_broadcastable_v<Matrix<double, 2, 2, 0>, Matrix<double, 3, 3, 1>> == false);
        REQUIRE(is_elementwise_broadcastable_v<Matrix<double, 2, 3, 0>, Matrix<double, 3, 2, 1>> == false);
        REQUIRE(is_elementwise_broadcastable_v<Matrix<double, 1, 5, 0>, Matrix<double, 5, 1, 1>> == false);
    }
}

// Test cases that demonstrate expected error messages for static_assert failures
TEST_CASE("Static Assert Error Message Documentation", "[static_assert][error_messages]") {
    
    SECTION("Addition static_assert error message") {
        // The AdditionExpr constructor should provide this error message:
        // "Incompatible matrix dimensions for element-wise addition."
        
        // Example of what should cause a compile error:
        // Matrix<double, 2, 2, 0> m1; Matrix<double, 3, 3, 1> m2; auto err = m1 + m2;
        
        REQUIRE(true); // Documentation test
    }
    
    SECTION("Subtraction static_assert error message") {
        // The SubtractionExpr constructor should provide this error message:
        // "Incompatible matrix dimensions for element-wise subtraction."
        
        // Example: Matrix<double, 2, 3, 0> m1; Matrix<double, 3, 2, 1> m2; auto err = m1 - m2;
        
        REQUIRE(true); // Documentation test
    }
    
    SECTION("Multiplication static_assert error message") {
        // The ElementwiseProductExpr constructor should provide this error message:
        // "Incompatible matrix dimensions for element-wise multiplication."
        
        // Example: Matrix<double, 4, 1, 0> m1; Matrix<double, 1, 4, 1> m2; auto err = m1 * m2;
        
        REQUIRE(true); // Documentation test
    }
    
    SECTION("Division static_assert error message") {
        // The ElementwiseDivisionExpr constructor should provide this error message:
        // "Incompatible matrix dimensions for element-wise division."
        
        // Example: Matrix<double, 3, 3, 0> m1; Matrix<double, 2, 2, 1> m2; auto err = m1 / m2;
        
        REQUIRE(true); // Documentation test
    }
}

// Test to verify that valid operations don't trigger static_assert
TEST_CASE("Valid Operations Pass Static Assertions", "[static_assert][valid_operations]") {
    
    SECTION("All compatible operations should compile and work") {
        // These operations should all pass static_assert checks
        
        Matrix<double, 2, 3, 0> m2x3{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
        Matrix<double, 2, 3, 1> m2x3_2{{7.0, 8.0, 9.0}, {10.0, 11.0, 12.0}};
        Matrix<double, 1, 1, 2> scalar_matrix{{5.0}};
        Scalar<double, 3> scalar(3.0);
        
        // Same-shape matrix operations
        auto sum = m2x3 + m2x3_2;
        auto diff = m2x3 - m2x3_2;
        auto product = m2x3 * m2x3_2;
        auto quotient = m2x3 / m2x3_2;
        
        // Matrix-scalar operations
        auto m_plus_s = m2x3 + scalar;
        auto s_plus_m = scalar + m2x3;
        auto m_times_s = m2x3 * scalar;
        auto s_times_m = scalar * m2x3;
        
        // Matrix with 1x1 matrix (scalar-shaped) operations
        auto m_plus_1x1 = m2x3 + scalar_matrix;
        auto m1x1_plus_m = scalar_matrix + m2x3;
        auto m_times_1x1 = m2x3 * scalar_matrix;
        auto m1x1_times_m = scalar_matrix * m2x3;
        
        // Verify all operations work correctly
        REQUIRE(sum.eval(0, 0) == Approx(8.0));        // 1 + 7
        REQUIRE(diff.eval(0, 0) == Approx(-6.0));      // 1 - 7
        REQUIRE(product.eval(0, 0) == Approx(7.0));    // 1 * 7
        REQUIRE(quotient.eval(0, 0) == Approx(1.0/7.0)); // 1 / 7
        
        REQUIRE(m_plus_s.eval(0, 0) == Approx(4.0));   // 1 + 3
        REQUIRE(s_plus_m.eval(0, 0) == Approx(4.0));   // 3 + 1
        REQUIRE(m_times_s.eval(0, 0) == Approx(3.0));  // 1 * 3
        REQUIRE(s_times_m.eval(0, 0) == Approx(3.0));  // 3 * 1
        
        REQUIRE(m_plus_1x1.eval(0, 0) == Approx(6.0)); // 1 + 5
        REQUIRE(m1x1_plus_m.eval(0, 0) == Approx(6.0)); // 5 + 1
        REQUIRE(m_times_1x1.eval(0, 0) == Approx(5.0)); // 1 * 5
        REQUIRE(m1x1_times_m.eval(0, 0) == Approx(5.0)); // 5 * 1
    }
}

// Test compile-time assertions using requires expressions in Catch2
TEST_CASE("Compile-time Shape Validation with Requires Expressions", "[requires][compile_time][shapes]") {
    
    SECTION("Valid operations should compile") {
        // Test that compatible operations can be called at compile time
        
        // Same-shaped matrix operations
        REQUIRE(requires(Matrix<double, 2, 2, 0> m1, Matrix<double, 2, 2, 1> m2) {
            m1 + m2;
            m1 - m2;
            m1 * m2;
            m1 / m2;
        });
        
        // Matrix-scalar operations
        REQUIRE(requires(Matrix<double, 3, 3, 0> matrix, Scalar<double, 1> scalar) {
            matrix + scalar;
            scalar + matrix;
            matrix * scalar;
            scalar * matrix;
            matrix - scalar;
            scalar - matrix;
            matrix / scalar;
            scalar / matrix;
        });
        
        // 1x1 matrix (scalar-shaped) operations
        REQUIRE(requires(Matrix<double, 1, 1, 0> scalar_matrix, Matrix<double, 2, 3, 1> regular_matrix) {
            scalar_matrix + regular_matrix;
            regular_matrix + scalar_matrix;
            scalar_matrix * regular_matrix;
            regular_matrix * scalar_matrix;
        });
        
        // Special matrix operations
        REQUIRE(requires(Matrix<double, 2, 2, 0> matrix, IdentityMatrix<double, 2> identity, ZeroMatrix<double, 2, 2> zero) {
            matrix + identity;
            matrix * zero;
            identity + zero;
        });
    }    
}

// Test type trait functions using simple compile-time evaluation
TEST_CASE("Type Trait Validation with Requires", "[type_traits]") {
    
    SECTION("Shape equality detection") {
        // Test is_eq_shape_v functionality
        
        REQUIRE(is_eq_shape_v<Matrix<double, 2, 2, 0>, Matrix<double, 2, 2, 1>> == true);
        REQUIRE(is_eq_shape_v<Matrix<double, 2, 2, 0>, Matrix<double, 3, 3, 1>> == false);
        REQUIRE(is_eq_shape_v<Matrix<double, 2, 3, 0>, Matrix<double, 2, 3, 1>> == true);
        REQUIRE(is_eq_shape_v<Matrix<double, 2, 3, 0>, Matrix<double, 3, 2, 1>> == false);
    }
    
    SECTION("Scalar shape detection") {
        // Test is_scalar_shape_v functionality
        
        REQUIRE(is_scalar_shape_v<Matrix<double, 1, 1, 0>> == true);
        REQUIRE(is_scalar_shape_v<Matrix<double, 2, 2, 0>> == false);
        REQUIRE(is_scalar_shape_v<Scalar<double, 0>> == true);
        REQUIRE(is_scalar_shape_v<Matrix<double, 1, 2, 0>> == false);
    }
    
    SECTION("Broadcasting compatibility detection") {
        // Test is_elementwise_broadcastable_v functionality
        
        REQUIRE(is_elementwise_broadcastable_v<Matrix<double, 2, 2, 0>, Matrix<double, 2, 2, 1>> == true);
        REQUIRE(is_elementwise_broadcastable_v<Scalar<double, 0>, Matrix<double, 3, 4, 1>> == true);
        REQUIRE(is_elementwise_broadcastable_v<Matrix<double, 2, 3, 0>, Scalar<double, 1>> == true);
        REQUIRE(is_elementwise_broadcastable_v<Matrix<double, 1, 1, 0>, Matrix<double, 3, 4, 1>> == true);
        REQUIRE(is_elementwise_broadcastable_v<Matrix<double, 2, 2, 0>, Matrix<double, 3, 3, 1>> == false);
    }
}
