//#include "TinyLA.h"
#include "TinyLA_ET.h"
#include <iostream>
#include <print>
#include <complex>
#include <format>
#include <sstream>
#include <vector>
#include <algorithm>

void print_expr(const auto& expr) {
    if constexpr (expr.rows == 1 && expr.cols == 1) {
        // Scalar case: simple one-line output
        std::println("{} = {}", expr.to_string(), expr.eval(0, 0));
    } else {
        // Matrix/Vector case: formatted multi-line output
        
        // First pass: calculate maximum width for each column
        std::vector<size_t> col_widths(expr.cols, 0);
        std::vector<std::vector<std::string>> formatted_values(expr.rows, std::vector<std::string>(expr.cols));
        
        for (uint32_t r = 0; r < expr.rows; ++r) {
            for (uint32_t c = 0; c < expr.cols; ++c) {
                auto value = expr.eval(r, c);
                
                // Format the value with appropriate precision
                std::string formatted;
                if constexpr (requires { value.real(); value.imag(); }) {
                    // Complex number
                    if (value.imag() >= 0) {
                        formatted = std::format("{:.6g}+{:.6g}i", value.real(), value.imag());
                    } else {
                        formatted = std::format("{:.6g}{:.6g}i", value.real(), value.imag());
                    }
                } else {
                    // Real number
                    formatted = std::format("{:.6g}", value);
                }
                
                formatted_values[r][c] = formatted;
                col_widths[c] = std::max(col_widths[c], formatted.length());
            }
        }
        
        // Calculate the expression name and equals sign offset
        std::string expr_str = expr.to_string();
        size_t max_name_length = 0;
        size_t current_line_length = 0;
        
        // Find the longest line in the expression string
        for (char c : expr_str) {
            if (c == '\n') {
                max_name_length = std::max(max_name_length, current_line_length);
                current_line_length = 0;
            } else {
                current_line_length++;
            }
        }
        // Don't forget the last line (if it doesn't end with \n)
        max_name_length = std::max(max_name_length, current_line_length);
        
        std::string indent(max_name_length + 3, ' '); // +3 for " = "
        
        // Print the matrix with proper alignment
        for (uint32_t r = 0; r < expr.rows; ++r) {
            if (r == 0) {
                // First row: include expression name
                std::print("{} = ", expr_str);
            } else {
                // Subsequent rows: indent to align with the opening bracket
                std::print("{}", indent);
            }
            
            // Choose bracket style based on matrix type and position
            char left_bracket, right_bracket;
            if (expr.rows == 1) {
                // Row vector - single row
                left_bracket = '['; right_bracket = ']';
            } else if (expr.cols == 1) {
                // Column vector - use vertical bars for multi-row
                left_bracket = '|'; right_bracket = '|';
            } else {
                // Matrix - use vertical bars for multi-row
                left_bracket = '|'; right_bracket = '|';
            }
            
            std::print("{}", left_bracket);
            
            for (uint32_t c = 0; c < expr.cols; ++c) {
                if (c > 0) std::print("  "); // Column separator
                
                // Right-align numbers in their column width
                std::print("{:>{}}", formatted_values[r][c], col_widths[c]);
            }
            
            std::println("{}", right_bracket);
        }
        std::println(); // Extra line for spacing
    }
}

int main() {
    TinyLA::Vector<double, 2, 0> s0{0.5};
    TinyLA::Scalar<double, 1> s1{6.0};
    TinyLA::Scalar<double, 2> s2{7.0};
    
    print_expr(s0);
    print_expr(s1);
    print_expr(s2);

    auto s_res = (s0 * s2 + s1) - 5;
    print_expr(s_res);
    auto s_res2 = derivate<0>(s_res);
    print_expr(s_res2);

    return 0;
}
