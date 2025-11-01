// Test file to verify static_assert compile-time errors
// This file should NOT compile due to static_assert failures

#include "include/TinyLA_ET.h"

using namespace TinyLA;

int main() {
    // This should cause a compile error: "Incompatible matrix dimensions for element-wise addition."
    Matrix<double, 2, 2, 0> m2x2{{1.0, 2.0}, {3.0, 4.0}};
    Matrix<double, 3, 3, 1> m3x3{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
    
    auto invalid_add = m2x2 + m3x3; // This should trigger static_assert
    
    return 0;
}